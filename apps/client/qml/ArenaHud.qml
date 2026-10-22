import QtQuick 2.15
import QtQuick.Layouts 1.15

Item {
    id: root
    // qmllint disable unqualified
    readonly property var stateContext: typeof arenaState !== "undefined" ? arenaState : null
    // qmllint enable unqualified
    property var player: stateContext ? stateContext.player : ({})
    property var robot: player.self || ({})
    property var display: player.display || ({})
    property var tactical: player.tactical || ({})
    property var arena: stateContext ? stateContext.arena : ({})
    property var teams: tactical.teams || ({})
    property var positions: tactical.positions || ({})
    property bool blocked: display.ui_blocked === true
    visible: stateContext ? stateContext.online : false

    function economy(team, key) {
        return Number((root.teams[team] || {})[key] || 0);
    }
    function invincible(team) {
        var p = root.positions[team + "_outpost"] || {};
        return p.controller === team && p.control === "full";
    }
    function buffsText(robot) {
        var b = robot.buffs || {}, values = [];
        if (b.invulnerable) values.push("无敌");
        if (b.defense_buff > 0) values.push("防御" + Math.round(b.defense_buff * 100) + "%");
        if (b.vulnerability > 0) values.push("易伤" + Math.round(b.vulnerability * 100) + "%");
        if (b.attack_multiplier > 1) values.push("攻击×" + Number(b.attack_multiplier).toFixed(1));
        if (b.cooling_multiplier > 1 || b.cooling_bonus > 0) values.push("冷却增益");
        if (b.posture) values.push(({attack:"进攻",defense:"防御",move:"移动"})[b.posture] || b.posture);
        return values.length ? values.join(" / ") : "—";
    }

    Item {
        anchors.fill: parent
        visible: !root.blocked
        Item {
            width: 1920; height: 170
            scale: root.width / 1920
            transformOrigin: Item.TopLeft
            TopInfoBar {
                anchors.fill: parent
                redBaseHP: Number(((root.tactical.bases || {}).red || {}).hp || 0)
                blueBaseHP: Number(((root.tactical.bases || {}).blue || {}).hp || 0)
                redOutpostHP: Number(((root.tactical.outposts || {}).red || {}).hp || 0)
                blueOutpostHP: Number(((root.tactical.outposts || {}).blue || {}).hp || 0)
                gameTime: Number((root.player.scene || {}).remaining || 0)
                redBaseMaxHP: 10000; blueBaseMaxHP: 10000
                redOutpostMaxHP: 1000; blueOutpostMaxHP: 1000
                redTeamName: "红方"; blueTeamName: "蓝方"
                redInvincible: root.invincible("red")
                blueInvincible: root.invincible("blue")
                redScore: root.economy("red", "control")
                blueScore: root.economy("blue", "control")
                redEconomy: root.economy("red", "resources")
                blueEconomy: root.economy("blue", "resources")
                showOutpostModule: true
            }
        }
        CentralAimingHUD {
            anchors.centerIn: parent
            scale: Math.min(1, root.width / 1500)
            visible: !root.display.camera_off && root.robot.role !== "aerial" && (root.player.scene || {}).status !== "finished"
            robot: root.robot
        }
        Rectangle {
            anchors.left: parent.left; anchors.bottom: parent.bottom
            anchors.margins: 16
            width: Math.min(370, root.width * .34); height: width * .57
            color: "#de0b1722"; border.color: "#426477"; radius: 5
            Canvas {
                id: map
                anchors.fill: parent; anchors.margins: 8
                onPaint: {
                    var ctx = getContext("2d"), fieldW = root.arena.width || 28, fieldH = root.arena.height || 15;
                    ctx.reset(); ctx.clearRect(0, 0, width, height);
                    ctx.save(); ctx.scale(width / fieldW, height / fieldH);
                    ctx.fillStyle = "#152a38"; ctx.fillRect(0, 0, fieldW, fieldH);
                    var walls = root.arena.obstacles || [];
                    ctx.fillStyle = "#567080";
                    for (var i = 0; i < walls.length; i++) {
                        var w = walls[i]; ctx.fillRect(w.x, w.y, w.w, w.h);
                    }
                    var zones = root.arena.zones || [], active = root.tactical.zones || {};
                    for (var j = 0; j < zones.length; j++) {
                        var z = zones[j], state = active[z.id] || {};
                        ctx.beginPath(); ctx.arc(z.x, z.y, z.radius || .3, 0, Math.PI * 2);
                        ctx.fillStyle = state.active ? "#88c76b" : "#344d5c"; ctx.fill();
                    }
                    var robots = root.tactical.robots || [];
                    for (var k = 0; k < robots.length; k++) {
                        var r = robots[k]; ctx.beginPath(); ctx.arc(r.x, r.y, r.id === root.robot.id ? .3 : .21, 0, Math.PI * 2);
                        ctx.fillStyle = !r.alive ? "#75838b" : r.team === "red" ? "#ff8977" : "#78cfff"; ctx.fill();
                    }
                    ctx.restore(); ctx.fillStyle = "#c8e3ef"; ctx.font = "11px sans-serif";
                    ctx.fillText("雷达解波 · 全场位置", 5, 13);
                    for (var n = 0; n < robots.length; n++) {
                        var point = robots[n]; ctx.fillText(point.slot, point.x / fieldW * width + 4, point.y / fieldH * height - 3);
                    }
                }
                Connections { target: root.stateContext; function onChanged() { map.requestPaint(); } }
            }
        }
        Rectangle {
            anchors.right: parent.right; anchors.bottom: parent.bottom; anchors.margins: 16
            width: Math.min(435, root.width * .42); height: radarRows.implicitHeight + 46
            color: "#de0b1722"; radius: 5; border.color: "#426477"
            Column {
                id: radarRows
                anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top; anchors.margins: 10
                spacing: 4
                Text { color: "#d3eaf4"; text: "雷达情报 · HP / 发弹量 / 增益 / 装甲受击"; font.pixelSize: 12 }
                Repeater {
                    model: root.tactical.robots || []
                    Text {
                        required property var modelData
                        width: radarRows.width
                        elide: Text.ElideRight
                        color: modelData.team === "red" ? "#ff9b8c" : "#8fd4ff"
                        font.pixelSize: Math.max(10, Math.min(12, root.width / 110))
                        text: (modelData.team === "red" ? "红" : "蓝") + modelData.slot + "  " + Math.ceil(modelData.hp) + "/" + Math.ceil(modelData.max_hp) + "  弹" + modelData.ammo + "  " + root.buffsText(modelData) + ((modelData.armor_hit_times || []).some(function(t) { return t >= (root.tactical.sample_time || 0) - 2; }) ? "  受击" : "")
                    }
                }
                Text { color: "#e5c4b5"; font.pixelSize: 11; text: "红方经济 " + root.economy("red", "resources") + " / 总 " + root.economy("red", "total_resources") }
                Text { color: "#b9def5"; font.pixelSize: 11; text: "蓝方经济 " + root.economy("blue", "resources") + " / 总 " + root.economy("blue", "total_resources") }
            }
        }
        Text {
            anchors.horizontalCenter: parent.horizontalCenter; anchors.bottom: parent.bottom; anchors.bottomMargin: 16
            color: "#d3edf6"; style: Text.Outline; styleColor: "#07121e"; font.pixelSize: 14
            text: "HP " + Math.ceil(root.robot.hp || 0) + "/" + Math.ceil(root.robot.max_hp || 0) + "   弹 " + (root.robot.ammo || 0) + "   热 " + Math.ceil(root.robot.heat || 0) + "/" + Math.ceil(root.robot.max_heat || 0)
        }
        Text {
            anchors.horizontalCenter: parent.horizontalCenter; anchors.top: parent.top; anchors.topMargin: root.width / 1920 * 155
            color: "#d3edf6"; font.pixelSize: 13
            text: "控制度 " + root.economy("red", "control").toFixed(1) + " : " + root.economy("blue", "control").toFixed(1) + "  · 后勤 " + Math.round((root.display.logistics || 1) * 100) + "%"
        }
        Rectangle {
            anchors.centerIn: parent
            width: Math.min(500, root.width * .6); height: 140; radius: 8
            color: "#ee101f2c"; border.color: "#6a9bb6"
            visible: (root.player.scene || {}).status === "finished"
            Column {
                anchors.centerIn: parent; spacing: 12
                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    color: "#e5f5ff"; font.pixelSize: 26
                    text: "比赛结束 · " + ((root.player.scene || {}).winner === "red" ? "红方胜利" : (root.player.scene || {}).winner === "blue" ? "蓝方胜利" : "平局")
                }
                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    color: "#bed8e8"; font.pixelSize: 15
                    text: "控制度 " + root.economy("red", "control").toFixed(1) + " : " + root.economy("blue", "control").toFixed(1) + " · 可保存回放或重新准备"
                }
            }
        }
    }
    Rectangle {
        anchors.fill: parent; color: "#050709"; visible: root.blocked
        Text { anchors.centerIn: parent; text: "判罚遮挡中"; color: "#c5d6df"; font.pixelSize: 24 }
    }
}
