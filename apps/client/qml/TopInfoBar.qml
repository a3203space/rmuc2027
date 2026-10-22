import QtQuick 2.15
import QtQuick.Layouts 1.15
Item {
    id: root
    property real redBaseHP: 0
    property real blueBaseHP: 0
    property real redBaseMaxHP: 10000
    property real blueBaseMaxHP: 10000
    property real redOutpostHP: 0
    property real blueOutpostHP: 0
    property real redOutpostMaxHP: 1000
    property real blueOutpostMaxHP: 1000
    property real redScore: 0
    property real blueScore: 0
    property int redEconomy: 0
    property int blueEconomy: 0
    property real gameTime: 0
    property bool redInvincible: false
    property bool blueInvincible: false
    property bool showOutpostModule: true
    property string redTeamName: "红方"
    property string blueTeamName: "蓝方"
    readonly property string formattedTime: Math.floor(Math.max(0, Math.ceil(gameTime)) / 60).toString().padStart(2, "0") + ":" + (Math.max(0, Math.ceil(gameTime)) % 60).toString().padStart(2, "0")
    Rectangle { anchors.fill: parent; color: "#d9101d29" }
    RowLayout {
        anchors.fill: parent; anchors.margins: 18; spacing: 24
        Repeater {
            model: ["red", "clock", "blue"]
            Item {
                required property string modelData
                Layout.fillWidth: true; Layout.fillHeight: true
                Column {
                    anchors.centerIn: parent; spacing: 7
                    Text { anchors.horizontalCenter: parent.horizontalCenter; color: modelData === "red" ? "#ff9b8c" : "#8fd4ff"; font.pixelSize: 25; text: modelData === "clock" ? "RMUC 2027 · 5v5" : modelData === "red" ? root.redTeamName : root.blueTeamName }
                    Text { anchors.horizontalCenter: parent.horizontalCenter; color: "#e6f4fc"; font.pixelSize: modelData === "clock" ? 42 : 27; text: modelData === "clock" ? root.formattedTime : "控制度 " + Number(modelData === "red" ? root.redScore : root.blueScore).toFixed(1) }
                    Text { anchors.horizontalCenter: parent.horizontalCenter; color: "#c5dce9"; font.pixelSize: 19; text: modelData === "clock" ? "重装 · 双步兵 · 机库 · 哨兵" : "基地 " + Math.ceil(modelData === "red" ? root.redBaseHP : root.blueBaseHP) + "/10000" + ((modelData === "red" ? root.redInvincible : root.blueInvincible) ? " · 保护" : "") }
                    Text { anchors.horizontalCenter: parent.horizontalCenter; color: "#a8c3d5"; font.pixelSize: 17; text: modelData === "clock" ? "" : "前哨 " + Math.ceil(modelData === "red" ? root.redOutpostHP : root.blueOutpostHP) + "/1000   经济 " + (modelData === "red" ? root.redEconomy : root.blueEconomy) }
                }
            }
        }
    }
}
