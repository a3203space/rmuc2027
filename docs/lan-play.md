# RMUC 2027 局域网游玩

构建步骤见根目录 [README](../README.md)。所有命令从项目根目录执行。

## 建房和加入

启动 `RMUC2027Client`。房主填写本工程根目录或准备好的运行包目录，点击“创建房间”；其他玩家填写房主局域网 IP 和端口 `8766`，点击“加入房间”。独立运行主机时，房主需填终端输出的 `owner_key`，普通玩家留空。

每人选择一个红/蓝方席位，步兵可在准备前选择血量/功率和冷却/爆发性能。全部真人准备后由房主开局，空位由 AI 补齐。

| 操作 | 输入 |
| --- | --- |
| 移动、瞄准、射击 | WASD、鼠标、左键 |
| 3 号轮腿 | W/S 前后，A/D 转向，鼠标独立控制云台 |
| 加速、跳跃、下蹲 | Shift、Space、Ctrl，按机器人能力执行 |
| 释放鼠标 | Esc |
| 重装 | 拾取、装配辅助、安装发射机构；手动移动取消辅助 |
| 机库 | 选择目标、起飞攻击、召回；不提供无人机相机 |
| 哨兵 | 驾驶射击、姿态切换；手动控制属于仿真游玩扩展 |
| 补给、回血、复活 | 右侧按钮，合法性与费用由主机判定 |

250 ms 无新输入时停止手动运动与射击；断线 3 秒后 AI 接管，席位保留 30 秒。重连需显式恢复控制。房主退出关闭房间。

双方雷达解波始终成功，地图和面板显示规则允许的战术情报；第一视角仍受视场和遮挡限制。过热断图传，基地受损降低图传清晰度，黄牌遮挡图传、地图和战术 UI。

## 回放

比赛结束后点击“保存本局回放”。本地查看：

```bash
uv run --locked rmuc2027 replay data/replays/某局.json
```

打开终端输出的本地地址。查看器支持暂停、拖动和自由观察；大回放由服务端建立帧索引、分批读取，不一次载入整局。页面可直接选择小于 64 MiB 的本地文件，更大的文件使用上述命令。

```bash
uv run --locked rmuc2027 verify-replay data/replays/某局.json
```

确定性校验要求记录回放的引擎版本与当前版本一致。历史 2027 回放仍可播放，跨版本不会被冒充为校验通过。

## 可搬移主机包

```bash
uv python install 3.12
uv run --locked python scripts/prepare_runtime.py --output build/runtime
# 已构建时可加 --skip-build
```

输出目录必须不存在。缓存完整时可使用 `--offline`；可加 `--installed-env .venv` 复用依赖并对照锁文件检查。运行包只包含同平台 Python、联机服务、引擎、固定补位 AI、规则和渲染资源，不包含研究工具、模型或历史数据。

Linux 执行 `build/runtime/host.sh`；Windows 执行 `build\runtime\host.cmd`。Windows 包必须在 Windows 准备，Linux 包必须在兼容的 Linux 准备。

Windows 客户端与同平台主机包一同部署：

```powershell
powershell -File scripts/build-client.ps1 -QtPrefix C:\Qt\6.8.3\msvc2022_64 -Output build\host-dist -Runtime build\runtime
```
