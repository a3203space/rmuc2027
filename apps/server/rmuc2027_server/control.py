"""Convert bounded player intent to ordinary SDK actions on the host."""
import math
from types import ModuleType

from rmuc2027_core.paths import ROOT
from rmuc2027_core import sdk

STAGES = {"positioning": "定位", "inserting": "插入", "lifting": "抬升", "flipping": "翻转",
          "rotating": "旋转", "verifying": "确认", "complete": "完成", "failed": "失败", "idle": "空闲"}
REASONS = {"heavy_absent": "重装未就位", "chassis_unstable": "底盘不稳定", "arm_out_of_reach": "机构未到达目标",
           "arm_misaligned": "机构方向未对齐", "heavy_destroyed": "重装机器人战亡", "pair_window": "双核心步骤未在时限内同步",
           "timeout": "装配超时", "left_assembly_area": "离开装配区"}


def baseline_helpers():
    # Only this bundled, fixed source is loaded; remote clients cannot supply code.
    path = ROOT / "packages/core/bots/default/strategy.py"
    module = ModuleType("arena_lan_helpers")
    source = path.read_text(encoding="utf-8").replace("from sdk import ", "from rmuc2027_core.sdk import ")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


class HumanController:
    def __init__(self, helpers):
        self.helpers = helpers
        self.state = {}
        self.mode = ""
        self.value = ""
        self.status = "手动控制"
        self.pending = []
        self.air_enabled = False
        self.air_target = "outpost"
        self.last_jump = False

    def operation(self, name, value, robot):
        if name in ("pick", "assemble", "install"):
            if robot["role"] != "heavy":
                raise ValueError("仅重装机器人支持该操作")
            if name == "assemble" and value not in ("1", "2", "3", "4"):
                raise ValueError("请选择 1–4 级装配难度")
            self.mode, self.value = name, value
            self.state["assist_started"] = False
            self.status = "辅助执行中"
        elif name == "cancel":
            self.mode = ""
            self.status = "已取消辅助，恢复手动控制"
        elif name in ("revive", "heal", "ammo", "posture"):
            if name == "posture" and (robot["role"] != "sentry" or value not in ("attack", "defense", "move")):
                raise ValueError("无效的哨兵姿态")
            if name == "ammo" and value not in ("local", "remote"):
                raise ValueError("请选择本地或远程补给")
            if len(self.pending) >= 8:
                raise ValueError("操作队列已满")
            self.pending.append((name, value))
        elif name.startswith("air_"):
            if robot["role"] != "aerial":
                raise ValueError("仅机库席位支持空中指令")
            if name == "air_target":
                if value not in ("outpost", "base"):
                    raise ValueError("请选择敌方前哨站或基地")
                self.air_target = value
                self.state.pop("fleet", None)
            else:
                self.air_enabled = name == "air_launch"
            self.status = "自动飞行：" + (("前哨站" if self.air_target == "outpost" else "基地") if self.air_enabled else "返航")

    def observe(self, frame, robot):
        for event in frame.get("events", []):
            if event.get("type") == "assembly_failed" and event.get("team") == robot["team"] and robot["role"] == "heavy":
                self.mode = ""
                self.status = "装配失败：" + REASONS.get(event.get("reason"), "请检查机构状态")

    def action(self, observation, robot, intent):
        h = self.helpers
        ctx = h.Context(observation, self.state)
        if robot["role"] == "aerial":
            return self._air(robot, ctx)
        forward, strafe = intent.forward, intent.strafe
        moving = math.hypot(forward, strafe) > .01
        if moving and self.mode:
            self.mode = ""
            self.status = "手动移动已取消辅助"
        if not robot["alive"]:
            self.mode = ""
            self.status = "等待复活"
        yaw = intent.yaw
        view_yaw = robot["aim"]
        size = max(1, math.hypot(forward, strafe))
        action = {"move": [(math.cos(view_yaw) * forward - math.sin(view_yaw) * strafe) / size,
                           (math.sin(view_yaw) * forward + math.cos(view_yaw) * strafe) / size],
                  "aim": yaw, "pitch": intent.pitch, "heading": yaw,
                  "fire": intent.fire, "skill": intent.boost, "crouch": intent.crouch,
                  "brake": not moving, "jump": 1.0 if intent.jump and not self.last_jump else 0.0}
        if robot.get("chassis_type") in ("wheel_legged", "serial_legged"):
            # Two-wheel legs cannot strafe. A/D steers the chassis while the
            # mouse continues to command the independent gimbal.
            action["move"] = [math.cos(robot["heading"]) * forward, math.sin(robot["heading"]) * forward]
            action["yaw_rate"] = strafe * 2.
        self.last_jump = intent.jump
        if self.mode and robot["alive"]:
            memory = self.state.setdefault("heavy", {})
            if self.mode == "pick":
                action = h.pick(robot, ctx)
                progress = robot.get("unit_pickup", {})
                self.status = "拾取能量单元 " + str(round(progress.get("progress", 0) * 100)) + "%" if progress else "前往能量单元拾取点"
                if robot.get("carried_units", 0) >= 2:
                    self.mode, self.status = "", "能量单元拾取完成"
                elif ctx.economy.get("energy_units_left", 0) == 0:
                    self.mode, self.status = "", "能量单元已耗尽"
            elif self.mode == "install":
                if robot.get("launcher") in ("fitted", "installed"):
                    self.mode, self.status = "", "发射机构安装完成"
                elif robot.get("launcher") not in ("unlocked", "installing"):
                    self.mode, self.status = "", "发射机构尚未解锁"
                else:
                    goal = sdk.workbench_point(robot, ctx.arena)
                    action = h.drive(robot, goal, observation, ctx.memory, .2) if h.distance(robot, goal) > .25 else {"move": [0, 0], "brake": True, "install_launcher": True}
                    self.status = "前往工作台 / 安装发射机构 " + str(round(robot.get("launcher_progress", 0) * 100)) + "%"
            elif self.mode == "assemble":
                difficulty = int(self.value)
                core = observation["assemblies"][ctx.team]
                if difficulty in ctx.economy.get("completed_difficulties", []):
                    self.mode, self.status = "", "装配完成"
                elif difficulty > 1 and difficulty - 1 not in ctx.economy.get("completed_difficulties", []):
                    self.mode, self.status = "", "请先完成前一级装配"
                elif core.get("permanent_lock"):
                    self.mode, self.status = "", "本级装配已被锁定"
                elif core["stage"] == "failed" and self.state.get("assist_started"):
                    self.mode, self.status = "", "装配失败：" + REASONS.get(core.get("blocked_reason"), "请检查状态后重试")
                elif core["stage"] in ("moving", "inserting"):
                    self.state["assist_started"] = True
                    action = h.assemble(robot, ctx, core, sdk.assembly_work_point(robot, ctx.arena))
                    self.status = "装配" + STAGES.get(core.get("operation"), "进行中") + " " + str(round(core.get("progress", 0) * 100)) + "%"
                    if core.get("blocked_reason"):
                        self.status += "：" + REASONS.get(core["blocked_reason"], "请检查机构状态")
                elif robot.get("carried_units", 0) < (2 if difficulty == 4 else 1):
                    if ctx.economy.get("energy_units_left", 0) == 0:
                        self.mode, self.status = "", "能量单元不足"
                    else:
                        action = h.pick(robot, ctx)
                        self.status = "前往拾取能量单元"
                else:
                    action = h.start_assembly(robot, ctx, difficulty, core, sdk.assembly_work_point(robot, ctx.arena), memory)
                    self.status = "前往装配区 / 等待等级解锁"
        for name, value in self.pending:
            if name in ("revive", "heal"):
                action["revive" if name == "revive" else "remote_heal"] = True
            elif name == "ammo":
                remote = value == "remote"
                action["buy_ammo"] = {"remote": remote, "amount": (10 if remote else 1) if robot["role"] == "heavy" else (100 if remote else 10)}
            elif name == "posture":
                action["posture"] = value
        self.pending.clear()
        return action

    def _air(self, robot, ctx):
        h = self.helpers
        fleet = robot.get("drones", [])
        target = dict(ctx.obs["outposts" if self.air_target == "outpost" else "bases"][ctx.enemy], spin=0.)
        if self.air_target == "outpost":
            h.track_spin(self.state, ctx.enemy, target, ctx.time)
            target["spin"] = self.state.get("spin", {}).get(ctx.enemy, 0)
        enabled = self.air_enabled and target["hp"] > 0
        memory = self.state.setdefault("fleet", {})
        hangar = ctx.arena["hangars"][ctx.team]
        commands = {}
        for k, drone in enumerate(fleet):
            command = {}
            if not enabled or drone["battery_wh"] < .5:
                if drone["state"] not in ("docked", "disabled"):
                    command = {"dock": True}
            elif drone["state"] in ("down", "docked"):
                command = {"launch": True}
            elif drone["state"] == "flying":
                exit_point = h.gantry_exit(hangar, target, ctx)
                state = memory.setdefault(drone["id"], {"mode": "exit"})
                if state["mode"] == "exit" and math.hypot(drone["x"] - exit_point[0], drone["y"] - exit_point[1]) > .2:
                    command = {"goto": [*exit_point, 2.]}
                else:
                    if state["mode"] == "exit":
                        state["mode"] = "approach"
                    command = h.ram(drone, target, k, len(fleet), ctx.time, memory, (hangar["x"], hangar["y"]))
            commands[str(drone["index"])] = command
        return {"drones": commands}
