import QtQuick 2.15
Item {
    id: root
    width: 600; height: 600
    property var robot: ({})
    readonly property real heatRatio: Number(robot.max_heat || 0) > 0 ? Math.min(1, Number(robot.heat || 0) / Number(robot.max_heat)) : 0
    readonly property bool isOverheated: robot.overheated === true
    readonly property bool centerSpeedWarning: robot.overspeed_locked === true || robot.overspeed_permanent_lock === true
    readonly property color heatColor: isOverheated ? "#ff4b4b" : heatRatio > .75 ? "#ffc17a" : "#bdeaff"
    Item {
        id: visualCenter
        anchors.centerIn: parent
        width: 1
        height: 1

        // --- 1. 外圈背景 (超大淡灰圆环) ---
        Image {
            id: outerRing
            anchors.centerIn: parent
            width: 560
            height: 560
            source: "qrc:/images/crosshair/crosshair_bg_single.png"
            fillMode: Image.PreserveAspectFit
            opacity: 0.35
        }

        // --- 横向基准线 ---
        Rectangle {
            anchors.verticalCenter: parent.verticalCenter
            anchors.right: heatRingContainer.left
            anchors.rightMargin: 4
            width: 178
            height: 1
            color: "#596067"
            opacity: 0.55
        }
        Rectangle {
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: heatRingContainer.right
            anchors.leftMargin: 4
            width: 178
            height: 1
            color: "#596067"
            opacity: 0.55
        }

        // --- 2. 热量环 (紧贴准星的小圈) ---
        Item {
            id: heatRingContainer
            anchors.centerIn: parent
            width: 116
            height: 116

            // 热量环背景
            Image {
                id: heatRingBg
                anchors.fill: parent
                source: "qrc:/images/crosshair/crosshair_slid_circle_bg.png"
                fillMode: Image.PreserveAspectFit
                opacity: 0.92
            }

            Canvas {
                id: heatProgressCanvas
                anchors.fill: parent
                antialiasing: true

                onPaint: {
                    var context = getContext("2d");
                    context.clearRect(0, 0, width, height);

                    var displayRatio = root.isOverheated ? 1.0 : root.heatRatio;
                    if (displayRatio <= 0) {
                        return;
                    }

                    var lineWidth = 8;
                    var radius = Math.max(0, Math.min(width, height) * 0.5 - lineWidth * 0.7);
                    var centerX = width * 0.5;
                    var centerY = height * 0.5;
                    var startAngle = -Math.PI * 0.5;
                    var endAngle = startAngle + Math.PI * 2 * displayRatio;

                    context.beginPath();
                    context.lineWidth = lineWidth;
                    context.lineCap = "round";
                    context.strokeStyle = root.heatColor;
                    context.shadowColor = root.heatColor;
                    context.shadowBlur = root.isOverheated ? 18 : 10;
                    context.arc(centerX, centerY, radius, startAngle, endAngle, false);
                    context.stroke();
                }

                Component.onCompleted: requestPaint()
            }

            Rectangle {
                anchors.centerIn: parent
                width: 94
                height: 94
                radius: 47
                color: "transparent"
                border.width: 1
                border.color: "#4D545B"
                opacity: 0.7
            }

            Text {
                id: overheatWarningText
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.bottom: parent.top
                anchors.bottomMargin: 10
                text: "热量超限，图传断开"
                visible: root.isOverheated
                color: "#FF4747"
                font.pixelSize: 18
                font.bold: true
                font.family: "PingFang SC"
                opacity: 0.92

                SequentialAnimation {
                    running: root.isOverheated
                    loops: Animation.Infinite
                    NumberAnimation {
                        target: overheatWarningText
                        property: "opacity"
                        to: 0.35
                        duration: 280
                    }
                    NumberAnimation {
                        target: overheatWarningText
                        property: "opacity"
                        to: 0.92
                        duration: 280
                    }
                }
            }

            Text {
                visible: root.centerSpeedWarning
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.top: parent.bottom
                anchors.topMargin: 10
                text: root.robot.overspeed_permanent_lock ? "发射机构永久锁定" : "发射机构超速锁定"
                color: "#FF3B3B"
                font.pixelSize: 14
                font.family: "PingFang SC"
                font.bold: true
            }
        }

        // --- 3. 准心 (中心十字) ---
        Image {
            id: crosshair
            anchors.centerIn: parent
            width: 36
            height: 36
            source: "qrc:/images/crosshair/crosshair_middle.png"
            fillMode: Image.PreserveAspectFit
            opacity: 1.0
            visible: !root.centerSpeedWarning
        }
    }

    onHeatRatioChanged: heatProgressCanvas.requestPaint()
    onHeatColorChanged: heatProgressCanvas.requestPaint()
    onIsOverheatedChanged: heatProgressCanvas.requestPaint()

    Column {
        x: parent.width / 2 + 286; y: parent.height / 2 - 28; spacing: 10
        Text { color: "#8e959c"; font.pixelSize: 12; text: "剩余发弹量" }
        Text { color: "#eef3f6"; font.pixelSize: 28; font.bold: true; text: Number(root.robot.ammo || 0) }
        Text { color: root.heatColor; font.pixelSize: 14; text: "热 " + Math.ceil(root.robot.heat || 0) + "/" + Math.ceil(root.robot.max_heat || 0) }
    }
}
