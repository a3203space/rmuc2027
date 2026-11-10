"""Display metadata for the three organizer-owned policies of each model and season."""

TACTICS = ("balanced", "assault", "guard")
# RMUC 2027 presets come from scripts/robot_policy_2027.py.
PROFILES_2027 = {
    "heavy": {
        "balanced": ("科技重炮", "取能量单元完成 1–4 级装配，装上 42mm 后从中央高地炮击建筑。"),
        "assault": ("速攻重炮", "完成 1–3 级装配后立即装炮，不做 4 级，尽早炮击敌方前哨与基地。"),
        "guard": ("守势重炮", "完成装配与装炮后守在己方前哨一侧，优先拦截来犯之敌。"),
    },
    "infantry": {
        "balanced": ("控制点争夺", "夺取敌方前哨控制点，180 秒后进攻敌方堡垒；己方控制点失守时回防。"),
        "assault": ("前哨突击", "直扑敌方前哨卡与中央高地，压制敌方前哨站。"),
        "guard": ("控制点守备", "守住己方中央高地与前哨卡，只拦截附近敌人，较早回补。"),
    },
    "sentry": {
        "balanced": ("中央高地哨卫", "占住己方中央高地卡守住前哨控制点，按战况切换进攻、防御、移动姿态。"),
        "assault": ("前压哨兵", "己方夺得敌方前哨控制点后前压到敌方中央高地卡。"),
        "guard": ("固守哨兵", "常驻己方中央高地卡，默认防御姿态。"),
    },
    "aerial": {
        "balanced": ("撞击编队", "空中机器人对准旋转装甲定时冲撞敌方前哨，之后撞击失去无敌的敌方基地。"),
        "assault": ("全程突袭", "持续撞击前哨与基地，不保留编队。"),
        "guard": ("前哨清障", "只撞击前哨站（含被对方重建为其所有的前哨），300 秒后才撞击基地。"),
    },
}


def preset_key(model, tactic, season="2027"):
    suffix = "" if tactic == "balanced" else "_" + tactic
    prefix = f"claude_opus55_xhigh_{season}_adaptive"
    return f"{prefix}_{model}{suffix}"


def preset_metadata(key, model, role, season=None):
    for current in ((season,) if season else ("2027",)):
        table = PROFILES_2027
        for tactic in TACTICS:
            if key == preset_key(model, tactic, current) and role in table:
                name, description = table[role][tactic]
                return {"id": tactic, "name": name, "description": description}
    return None
