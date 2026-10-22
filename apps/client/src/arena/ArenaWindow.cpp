#include "ArenaWindow.h"
#include "ArenaConnection.h"
#include "ArenaStateAdapter.h"

#include <QApplication>
#include <QCloseEvent>
#include <QComboBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFile>
#include <QDebug>
#include <QElapsedTimer>
#include <memory>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QNetworkInterface>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QQmlContext>
#include <QQuickWidget>
#include <QScrollArea>
#include <QSettings>
#include <QSlider>
#include <QSpinBox>
#include <QStandardPaths>
#include <QTimer>
#include <QVBoxLayout>
#include <QWebChannel>
#include <QWebEnginePage>
#include <QWebEngineSettings>
#include <QWebEngineScript>
#include <QWebEngineScriptCollection>
#include <QWebEngineView>

namespace RM {

ArenaWindow::ArenaWindow(QWidget *parent) : QMainWindow(parent) {
  setWindowTitle(QStringLiteral("RMUC 2027 局域网仿真"));
  resize(1600, 900); setMinimumSize(1280, 720);
  m_connection = new ArenaConnection(this);
  m_state = new ArenaStateAdapter(this);
  m_bridge = new ArenaBridge(this);
  auto *central = new QWidget(this);
  auto *layout = new QHBoxLayout(central); layout->setContentsMargins(0, 0, 0, 0); layout->setSpacing(0);
  setCentralWidget(central);
  auto *viewport = new QWidget(central);
  auto *stack = new QGridLayout(viewport); stack->setContentsMargins(0, 0, 0, 0);
  m_view = new QWebEngineView(viewport);
  m_view->setContextMenuPolicy(Qt::NoContextMenu);
  m_view->setHtml(QStringLiteral("<body style='background:#08131e;color:#bad9e9;display:grid;place-items:center;height:95vh;font-family:sans-serif'>创建或加入 Arena 房间</body>"));
  m_view->settings()->setAttribute(QWebEngineSettings::JavascriptCanOpenWindows, false);
  auto *channel = new QWebChannel(m_view->page()); channel->registerObject("arena", m_bridge);
  m_view->page()->setWebChannel(channel);
  QFile channelSource(QStringLiteral(":/qtwebchannel/qwebchannel.js"));
  if (channelSource.open(QIODevice::ReadOnly)) {
    QWebEngineScript script; script.setName("arena-webchannel");
    script.setInjectionPoint(QWebEngineScript::DocumentCreation);
    script.setWorldId(QWebEngineScript::MainWorld);
    script.setSourceCode(QString::fromUtf8(channelSource.readAll()));
    m_view->page()->scripts().insert(script);
  }
  connect(m_view->page(), &QWebEnginePage::featurePermissionRequested, this,
          [this](const QUrl &origin, QWebEnginePage::Feature feature) {
    const bool same = origin.host() == m_connection->serverUrl().host() && origin.port(80) == m_connection->serverUrl().port(80);
    m_view->page()->setFeaturePermission(origin, feature, same && feature == QWebEnginePage::MouseLock ? QWebEnginePage::PermissionGrantedByUser : QWebEnginePage::PermissionDeniedByUser);
  });
  stack->addWidget(m_view, 0, 0);
  m_hud = new QQuickWidget(viewport);
  m_hud->setResizeMode(QQuickWidget::SizeRootObjectToView);
  m_hud->setClearColor(Qt::transparent);
  m_hud->setAttribute(Qt::WA_AlwaysStackOnTop);
  m_hud->setAttribute(Qt::WA_TranslucentBackground);
  m_hud->setAttribute(Qt::WA_TransparentForMouseEvents);
  m_hud->rootContext()->setContextProperty("arenaState", m_state);
  m_hud->setSource(QUrl("qrc:/qml/ArenaHud.qml"));
  stack->addWidget(m_hud, 0, 0);
  layout->addWidget(viewport, 1);

  auto *scroll = new QScrollArea(central); scroll->setWidgetResizable(true); scroll->setMinimumWidth(300); scroll->setMaximumWidth(370);
  auto *panel = new QWidget(scroll); auto *side = new QVBoxLayout(panel); side->setSpacing(12);
  scroll->setWidget(panel); layout->addWidget(scroll);
  auto *title = new QLabel(QStringLiteral("ARENA 2027\n局域网仿真"), panel); title->setStyleSheet("font-size:23px;font-weight:700;color:#cbeaff;padding:10px 0;"); side->addWidget(title);
  auto *form = new QFormLayout;
  QSettings settings;
  m_nickname = new QLineEdit(settings.value("arena/nickname", "玩家").toString(), panel); m_nickname->setMaxLength(24);
  m_address = new QLineEdit(settings.value("arena/host", "127.0.0.1").toString(), panel);
  m_port = new QSpinBox(panel); m_port->setRange(1024, 65535); m_port->setValue(settings.value("arena/port", 8766).toInt());
  m_runtime = new QLineEdit(runtimeRoot(), panel);
  m_ownerKey = new QLineEdit(panel); m_ownerKey->setEchoMode(QLineEdit::Password);
  m_ownerKey->setPlaceholderText(QStringLiteral("独立主机房主填写，普通玩家留空"));
  m_runtime->setPlaceholderText(QStringLiteral("RMUC 2027 工程或运行包目录"));
  form->addRow(QStringLiteral("昵称"), m_nickname); form->addRow(QStringLiteral("主机 IP"), m_address); form->addRow(QStringLiteral("端口"), m_port);
  form->addRow(QStringLiteral("仿真目录"), m_runtime); side->addLayout(form);
  form->addRow(QStringLiteral("房主口令"), m_ownerKey);
  auto *buttons = new QHBoxLayout;
  auto *hostButton = new QPushButton(QStringLiteral("创建房间"), panel), *joinButton = new QPushButton(QStringLiteral("加入房间"), panel);
  buttons->addWidget(hostButton); buttons->addWidget(joinButton); side->addLayout(buttons);
  connect(hostButton, &QPushButton::clicked, this, &ArenaWindow::host);
  connect(joinButton, &QPushButton::clicked, this, [this] { join(); });
  m_roomLabel = new QLabel(QStringLiteral("空位由 AI 补齐 · 5 对 5"), panel); m_roomLabel->setWordWrap(true); side->addWidget(m_roomLabel);
  auto *seats = new QGridLayout;
  const QList<int> slots{1, 3, 4, 6, 7}; const QStringList roles{QStringLiteral("重装"), QStringLiteral("步兵"), QStringLiteral("步兵"), QStringLiteral("机库"), QStringLiteral("哨兵")};
  seats->addWidget(new QLabel(QStringLiteral("红方")), 0, 0); seats->addWidget(new QLabel(QStringLiteral("蓝方")), 0, 1);
  for (int row = 0; row < slots.size(); ++row) for (int column = 0; column < 2; ++column) {
    const QString team = column == 0 ? "red" : "blue"; const int slot = slots[row];
    const QString key = team + ":" + QString::number(slot);
    auto *button = new QPushButton(QString::number(slot) + " " + roles[row] + " · AI", panel);
    button->setCheckable(true); m_seats.insert(key, button); seats->addWidget(button, row + 1, column);
    connect(button, &QPushButton::clicked, this, [this, team, slot] { m_connection->request("seat", {{"team", team}, {"slot", slot}}); });
  }
  side->addLayout(seats);
  m_ready = new QPushButton(QStringLiteral("准备"), panel); m_start = new QPushButton(QStringLiteral("开始比赛 / 再来一局"), panel);
  side->addWidget(m_ready); side->addWidget(m_start);
  connect(m_ready, &QPushButton::clicked, this, [this] { m_connection->request("ready", {{"ready", !m_isReady}}); });
  connect(m_start, &QPushButton::clicked, this, [this] { m_connection->request("start"); });
  auto *resume = new QPushButton(QStringLiteral("重连后恢复控制"), panel); side->addWidget(resume);
  connect(resume, &QPushButton::clicked, this, [this] { m_connection->operation("resume"); m_view->setFocus(); });
  m_robotLabel = new QLabel(panel); m_robotLabel->setWordWrap(true); side->addWidget(m_robotLabel);
  auto addAction = [this, panel, side](const QString &label, const QString &name, std::function<QString()> value = {}) {
    auto *button = new QPushButton(label, panel); side->addWidget(button);
    m_actions.insert(name + (name == "ammo" ? label : QString()), button);
    button->setEnabled(false);
    connect(button, &QPushButton::clicked, this, [this, name, value] {
      const QString argument = value ? value() : QString();
      if (name == "profile") {
        m_connection->request("profile", {{"chassis", argument.section(':', 0, 0)}, {"weapon", argument.section(':', 1, 1)}});
      } else m_connection->operation(name, argument);
    });
    return button;
  };
  auto *special = new QGroupBox(QStringLiteral("机构与补给"), panel); side->addWidget(special);
  auto *specialLayout = new QFormLayout(special);
  m_difficulty = new QComboBox(special); m_difficulty->addItems({"1", "2", "3", "4"}); specialLayout->addRow(QStringLiteral("装配难度"), m_difficulty);
  m_posture = new QComboBox(special); m_posture->addItem(QStringLiteral("进攻"), "attack"); m_posture->addItem(QStringLiteral("防御"), "defense"); m_posture->addItem(QStringLiteral("移动"), "move"); specialLayout->addRow(QStringLiteral("哨兵姿态"), m_posture);
  m_airTarget = new QComboBox(special); m_airTarget->addItem(QStringLiteral("敌方前哨站"), "outpost"); m_airTarget->addItem(QStringLiteral("敌方基地"), "base"); specialLayout->addRow(QStringLiteral("空中目标"), m_airTarget);
  m_profile = new QComboBox(special);
  m_profile->addItem(QStringLiteral("血量 / 冷却"), "hp:cooling"); m_profile->addItem(QStringLiteral("血量 / 爆发"), "hp:burst"); m_profile->addItem(QStringLiteral("功率 / 冷却"), "power:cooling"); m_profile->addItem(QStringLiteral("功率 / 爆发"), "power:burst"); specialLayout->addRow(QStringLiteral("步兵性能"), m_profile);
  addAction(QStringLiteral("拾取能量单元"), "pick");
  addAction(QStringLiteral("启动装配辅助"), "assemble", [this] { return m_difficulty->currentText(); });
  addAction(QStringLiteral("取消辅助"), "cancel"); addAction(QStringLiteral("安装发射机构"), "install");
  addAction(QStringLiteral("切换哨兵姿态"), "posture", [this] { return m_posture->currentData().toString(); });
  addAction(QStringLiteral("选择空中目标"), "air_target", [this] { return m_airTarget->currentData().toString(); });
  addAction(QStringLiteral("起飞并攻击"), "air_launch"); addAction(QStringLiteral("空中系统返航"), "air_return");
  addAction(QStringLiteral("选择步兵性能"), "profile", [this] { return m_profile->currentData().toString(); });
  addAction(QStringLiteral("补给区购买弹药"), "ammo", [] { return "local"; });
  addAction(QStringLiteral("远程购买弹药"), "ammo", [] { return "remote"; });
  addAction(QStringLiteral("远程回血"), "heal"); addAction(QStringLiteral("购买立即复活"), "revive");
  auto *sensitivity = new QSlider(Qt::Horizontal, panel); sensitivity->setRange(1, 50); sensitivity->setValue(settings.value("arena/sensitivity", 20).toInt());
  side->addWidget(new QLabel(QStringLiteral("鼠标灵敏度"), panel)); side->addWidget(sensitivity);
  connect(sensitivity, &QSlider::valueChanged, this, [this](int value) { QSettings().setValue("arena/sensitivity", value); m_view->page()->runJavaScript(QString("localStorage.setItem('arena-sensitivity',%1)").arg(value * .0001)); });
  auto *save = new QPushButton(QStringLiteral("保存本局回放"), panel); side->addWidget(save);
  connect(save, &QPushButton::clicked, this, [this] {
    neutral(); const QString path = QFileDialog::getSaveFileName(this, QStringLiteral("保存 RMUC 2027 回放"), "rmuc2027-replay.json", "RMUC 2027 replay (*.json)");
    if (!path.isEmpty()) m_connection->downloadReplay(path);
  });
  auto *leaveButton = new QPushButton(QStringLiteral("离开房间"), panel); side->addWidget(leaveButton);
  connect(leaveButton, &QPushButton::clicked, this, [this] { neutral(); m_connection->leave(); stopHost(); m_state->apply({}); });
  auto *hint = new QLabel(QStringLiteral("WASD 移动 · 鼠标瞄准 · 左键射击\nShift 加速 · 空格跳跃 · Ctrl 下蹲\nEsc 释放鼠标\n哨兵手动控制为仿真游玩扩展。\n雷达解波始终成功，图传遵循视场和遮挡。"), panel); hint->setWordWrap(true); side->addWidget(hint);
  m_status = new QLabel(panel); m_status->setWordWrap(true); side->addWidget(m_status); side->addStretch();
  panel->setStyleSheet("QWidget{background:#101d29;color:#cadde8;font-size:13px;} QLineEdit,QSpinBox,QComboBox{background:#182b3b;padding:6px;border:1px solid #30495b;border-radius:4px;} QPushButton{background:#203d51;border:1px solid #375e76;padding:8px;border-radius:4px;} QPushButton:hover{background:#2b5068;} QPushButton:checked{background:#346278;} QPushButton:disabled{color:#677986;} QGroupBox{border:1px solid #30495b;margin-top:14px;padding-top:12px;} QGroupBox::title{subcontrol-origin:margin;left:9px;}");
  connect(m_connection, &ArenaConnection::roomChanged, this, &ArenaWindow::updateRoom);
  connect(m_connection, &ArenaConnection::snapshotReceived, this, &ArenaWindow::onSnapshot);
  connect(m_connection, &ArenaConnection::error, m_status, &QLabel::setText);
  connect(m_connection, &ArenaConnection::connectionChanged, this, [this](bool connected) { if (!connected) neutral(); m_status->setText(connected ? QStringLiteral("已连接仿真主机") : QStringLiteral("连接中断，正在尝试重连")); });
  connect(m_bridge, &ArenaBridge::inputReceived, this, [this](const QString &payload) {
    m_lastInput = QJsonDocument::fromJson(payload.toUtf8()).object();
    if (!m_lastInput.isEmpty()) m_connection->sendInput(m_lastInput);
  });
  connect(m_bridge, &ArenaBridge::ready, this, [this] {
    emit m_bridge->snapshot(QString::fromUtf8(QJsonDocument(m_lastSnapshot).toJson(QJsonDocument::Compact)));
    const double sensitivity = QSettings().value("arena/sensitivity", 20).toDouble() * .0001;
    m_view->page()->runJavaScript(QString("localStorage.setItem('arena-sensitivity',%1)").arg(sensitivity));
  });
  connect(m_bridge, &ArenaBridge::paused, this, &ArenaWindow::neutral);
  auto frameLogClock = std::make_shared<QElapsedTimer>(); frameLogClock->start();
  connect(m_bridge, &ArenaBridge::renderStatsReceived, this, [frameLogClock](const QString &payload) {
    if (frameLogClock->elapsed() < 900 || payload.size() > 1024) return;
    const auto stats = QJsonDocument::fromJson(payload.toUtf8()).object();
    qInfo() << "ARENA_RENDER_TIMING fps=" << stats.value("fps").toDouble()
            << "tick=" << stats.value("tick").toInt();
    frameLogClock->restart();
  });
  connect(m_view, &QWebEngineView::loadFinished, this, [this](bool ok) { if (!ok) m_status->setText(QStringLiteral("仿真画面加载失败，请确认主机已构建前端资源")); });
  connect(&m_host, &QProcess::readyReadStandardOutput, this, [this] {
    m_hostOutput += m_host.readAllStandardOutput();
    while (m_hostOutput.contains('\n')) {
      const int end = m_hostOutput.indexOf('\n'); const QByteArray line = m_hostOutput.left(end); m_hostOutput.remove(0, end + 1);
      const QJsonObject ready = QJsonDocument::fromJson(line).object();
      if (ready.value("lan_ready").toBool()) {
        m_address->setText("127.0.0.1");
        QStringList addresses;
        for (const QHostAddress &address : QNetworkInterface::allAddresses())
          if (address.protocol() == QAbstractSocket::IPv4Protocol && !address.isLoopback()) addresses.append(address.toString());
        if (addresses.isEmpty()) for (const QJsonValue &value : ready.value("addresses").toArray()) addresses.append(value.toString());
        addresses.removeDuplicates();
        m_localAddresses = addresses.join(" / ") + ":" + QString::number(m_port->value());
        m_status->setText(QStringLiteral("房主地址：%1:%2").arg(addresses.join(" / ")).arg(m_port->value()));
        join(ready.value("owner_key").toString());
      }
    }
  });
  connect(&m_host, &QProcess::readyReadStandardError, this, [this] {
    const QString text = QString::fromUtf8(m_host.readAllStandardError()).trimmed();
    if (!text.isEmpty()) m_status->setText(text.right(700));
  });
  connect(&m_host, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) { m_status->setText(QStringLiteral("启动仿真主机失败：%1").arg(m_host.errorString())); });
  connect(&m_host, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int code, QProcess::ExitStatus) { if (!m_stopping) m_status->setText(QStringLiteral("仿真主机已退出（%1）").arg(code)); });
  qApp->installEventFilter(this);
}

QString ArenaWindow::runtimeRoot() const {
  const QString env = qEnvironmentVariable("RMUC_ARENA_ROOT");
  if (!env.isEmpty()) return env;
  const QString configured = QSettings().value("arena/runtime").toString();
  if (!configured.isEmpty()) return configured;
  for (const QString &origin : {QCoreApplication::applicationDirPath(), QDir::currentPath()}) {
    QDir parent(origin);
    for (int depth = 0; depth < 6; ++depth) {
      for (const QString &suffix : {QString(), QStringLiteral("/runtime/app")}) {
        const QString path = parent.absolutePath() + suffix;
        if (QFileInfo::exists(path + "/apps/server/rmuc2027_server/server.py")) return path;
      }
      if (!parent.cdUp()) break;
    }
  }
  return {};
}

void ArenaWindow::host() {
  if (m_host.state() != QProcess::NotRunning) { m_status->setText(QStringLiteral("本机房间已在运行")); return; }
  if (m_connection->hasSession()) { m_status->setText(QStringLiteral("请先离开当前房间，再创建房间")); return; }
  QDir root(m_runtime->text().trimmed());
  if (QFileInfo::exists(root.filePath("app/apps/server/rmuc2027_server/server.py"))) root.cd("app");
  if (!QFileInfo::exists(root.filePath("apps/server/rmuc2027_server/server.py")) || !QFileInfo::exists(root.filePath("build/web/apps/client/web/index.html"))) {
    m_status->setText(QStringLiteral("仿真目录不完整，请先执行局域网运行包准备脚本")); return;
  }
#ifdef Q_OS_WIN
  const QStringList interpreters{root.filePath("../python/python.exe"), root.filePath(".venv/Scripts/python.exe")};
  const QString engine = root.filePath("target/release/rmuc-sim.exe");
#else
  const QStringList interpreters{root.filePath("../python/bin/python3"), root.filePath(".venv/bin/python")};
  const QString engine = root.filePath("target/release/rmuc-sim");
#endif
  QString python = qEnvironmentVariable("RMUC_PYTHON");
  if (python.isEmpty()) for (const QString &path : interpreters) if (QFileInfo::exists(path)) { python = path; break; }
  if (python.isEmpty() || !QFileInfo::exists(engine)) { m_status->setText(QStringLiteral("缺少 Python 运行环境或 Rust 仿真程序，请先运行准备脚本")); return; }
  QSettings().setValue("arena/runtime", root.absolutePath());
  m_host.setWorkingDirectory(root.absolutePath()); m_hostOutput.clear(); m_stopping = false;
  auto environment = QProcessEnvironment::systemEnvironment(); environment.insert("PYTHONUTF8", "1"); environment.insert("PYTHONDONTWRITEBYTECODE", "1");
  environment.insert("RMUC_RESOURCE_ROOT", root.absolutePath());
  environment.insert("PYTHONPATH", root.filePath("packages/core") + QDir::listSeparator() + root.filePath("apps/server"));
  environment.insert("RMUC_USER_DATA", QDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation)).filePath("arena"));
  m_host.setProcessEnvironment(environment);
  m_host.start(python, {"-u", "-m", "rmuc2027_server", "host", "--port", QString::number(m_port->value()), "--parent-stdin"});
  m_status->setText(QStringLiteral("正在启动仿真主机…"));
}

void ArenaWindow::join(const QString &key) {
  QSettings().setValue("arena/nickname", m_nickname->text().trimmed()); QSettings().setValue("arena/host", m_address->text().trimmed());
  QUrl url; url.setScheme("http"); url.setHost(m_address->text().trimmed()); url.setPort(m_port->value());
  if (!m_connection->join(url, m_nickname->text().trimmed(), key.isEmpty() ? m_ownerKey->text() : key)) return;
  QSettings().setValue("arena/port", m_port->value());
  const QString page = url.resolved(QUrl("/play")).toString();
  if (page != m_loadedUrl) { m_loadedUrl = page; m_view->load(QUrl(page)); }
}

void ArenaWindow::updateRoom(const QJsonObject &room) {
  const QString status = room.value("status").toString();
  const QMap<QString, QString> phases{{"lobby", QStringLiteral("大厅")}, {"starting", QStringLiteral("开赛准备")}, {"running", QStringLiteral("对局中")}, {"settling", QStringLiteral("保存回放")}, {"finished", QStringLiteral("已结束")}, {"closed", QStringLiteral("已关闭")}};
  m_roomLabel->setText(QStringLiteral("2027 · %1 · 空位 AI 补齐").arg(phases.value(status, status)));
  if (!m_localAddresses.isEmpty()) m_roomLabel->setText(m_roomLabel->text() + "\n" + QStringLiteral("房主地址：") + m_localAddresses);
  QJsonObject me;
  for (const QJsonValue &entry : room.value("players").toArray()) if (entry.toObject().value("id").toString() == m_playerId) me = entry.toObject();
  m_isReady = me.value("ready").toBool(); m_ready->setText(m_isReady ? QStringLiteral("取消准备") : QStringLiteral("准备"));
  m_ready->setEnabled((status == "lobby" || status == "finished") && !me.value("slot").isNull());
  m_start->setEnabled(me.value("host").toBool() && (status == "lobby" || status == "finished"));
  if (m_actions.contains("profile")) m_actions["profile"]->setEnabled((status == "lobby" || status == "finished") && (me.value("slot").toInt() == 3 || me.value("slot").toInt() == 4));
  for (auto it = m_seats.begin(); it != m_seats.end(); ++it) {
    const QString team = it.key().section(':', 0, 0); const int slot = it.key().section(':', 1).toInt();
    QJsonObject occupant;
    for (const QJsonValue &entry : room.value("players").toArray()) { const auto p = entry.toObject(); if (p.value("team").toString() == team && p.value("slot").toInt() == slot) occupant = p; }
    const bool own = occupant.value("id").toString() == m_playerId && !occupant.isEmpty();
    const QString role = slot == 1 ? QStringLiteral("重装") : slot == 6 ? QStringLiteral("机库") : slot == 7 ? QStringLiteral("哨兵") : QStringLiteral("步兵");
    it.value()->setText(QString::number(slot) + " " + role + " · " + (occupant.isEmpty() ? "AI" : occupant.value("nickname").toString()) + (!occupant.isEmpty() && !occupant.value("connected").toBool() ? QStringLiteral("（预留）") : QString()));
    it.value()->setChecked(own);
    it.value()->setEnabled((occupant.isEmpty() || own) && (status == "lobby" || status == "finished" || status == "running") && !(status == "running" && !me.value("slot").isNull() && !own));
  }
  if (!room.value("error").isNull()) m_status->setText(room.value("error").toString());
  const auto result = room.value("result").toObject();
  if (!result.isEmpty()) {
    const QString winner = result.value("winner").toString();
    m_roomLabel->setText(QStringLiteral("比赛结束 · %1").arg(winner == "red" ? QStringLiteral("红方胜利") : winner == "blue" ? QStringLiteral("蓝方胜利") : QStringLiteral("平局")));
    if (!m_localAddresses.isEmpty()) m_roomLabel->setText(m_roomLabel->text() + "\n" + QStringLiteral("房主地址：") + m_localAddresses);
  }
}

void ArenaWindow::onSnapshot(const QJsonObject &snapshot) {
  if (snapshot.contains("player_id")) m_playerId = snapshot.value("player_id").toString();
  // Retain static assets for the late WebChannel readiness handshake.
  const auto oldStatic = m_lastSnapshot.value("static");
  m_lastSnapshot = snapshot;
  if (!snapshot.contains("static") && oldStatic.isObject()) m_lastSnapshot.insert("static", oldStatic);
  m_state->apply(m_lastSnapshot);
  if (snapshot.contains("room")) updateRoom(snapshot.value("room").toObject());
  emit m_bridge->snapshot(QString::fromUtf8(QJsonDocument(snapshot).toJson(QJsonDocument::Compact)));
  const QJsonObject p = snapshot.value("player").toObject(), own = p.value("self").toObject();
  m_robotLabel->setText(p.isEmpty() ? QString() : QStringLiteral("%1 %2 号 · HP %3/%4\n%5").arg(p.value("team").toString()).arg(p.value("slot").toInt()).arg(own.value("hp").toInt()).arg(own.value("max_hp").toInt()).arg(p.value("control").toObject().value("status").toString()));
  const QString role = own.value("role").toString();
  if (own.value("chassis_type").toString() == "wheel_legged")
    m_robotLabel->setText(m_robotLabel->text() + QStringLiteral("\n轮腿底盘：W/S 前后，A/D 转向，鼠标独立控制云台"));
  if (role == "heavy" && own.contains("hp")) {
    const QMap<QString, QString> launchers{{"locked", QStringLiteral("未解锁")}, {"unlocked", QStringLiteral("可安装")}, {"installing", QStringLiteral("安装中")}, {"fitted", QStringLiteral("待首次发射确认")}, {"installed", QStringLiteral("已安装")}};
    m_robotLabel->setText(m_robotLabel->text() + QStringLiteral("\n携带能量单元 %1 · 发射机构 %2").arg(own.value("carried_units").toInt()).arg(launchers.value(own.value("launcher").toString())));
  }
  if (role == "aerial" && own.contains("hp")) {
    const QMap<QString, QString> states{{"docked", QStringLiteral("停靠")}, {"flying", QStringLiteral("飞行")}, {"down", QStringLiteral("待回收")}, {"disabled", QStringLiteral("不可用")}};
    for (const QJsonValue &entry : p.value("air").toArray()) {
      const auto drone = entry.toObject();
      m_robotLabel->setText(m_robotLabel->text() + QStringLiteral("\n无人机 %1 · %2 · %3 Wh").arg(drone.value("index").toInt()).arg(states.value(drone.value("state").toString())).arg(drone.value("battery_wh").toDouble(), 0, 'f', 1));
    }
  }
  if (own.contains("hp") && !own.value("alive").toBool() && role != "aerial")
    m_robotLabel->setText(m_robotLabel->text() + QStringLiteral("\n复活进度 %1 / %2").arg(own.value("respawn_progress").toDouble(), 0, 'f', 1).arg(own.value("respawn_required").toDouble(), 0, 'f', 1));
  const bool canAct = snapshot.value("room").toObject().value("status").toString() == "running" && !p.isEmpty() && !p.value("display").toObject().value("ui_blocked").toBool() && !own.value("red_card").toBool() && !p.value("control").toObject().value("needs_resume").toBool();
  for (auto it = m_actions.begin(); it != m_actions.end(); ++it) {
    const QString action = it.key();
    if (action == "profile") continue; // Prepared in the lobby and locked at tick zero.
    const bool allowed = action.startsWith("air_") ? role == "aerial" : action == "posture" ? role == "sentry" :
                         action == "profile" ? role == "infantry" :
                         (action == "pick" || action == "assemble" || action == "install" || action == "cancel") ? role == "heavy" : role != "aerial";
    it.value()->setEnabled(canAct && allowed);
  }
  if (p.value("display").toObject().value("ui_blocked").toBool()) {
    m_robotLabel->setText(QStringLiteral("判罚遮挡中"));
    // Native controls carry no tactical state while the HUD is blocked.
  }
  for (const auto &value : snapshot.value("operations").toArray()) {
    const auto operation = value.toObject();
    if (operation.value("status").toString() == "rejected") m_status->setText(operation.value("message").toString());
  }
}

void ArenaWindow::neutral() {
  QJsonObject input{{"yaw", m_lastInput.value("yaw").toDouble()}, {"pitch", m_lastInput.value("pitch").toDouble()}, {"active", false}};
  m_connection->sendInput(input);
  m_connection->operation("cancel");
  m_view->page()->runJavaScript("document.exitPointerLock();window.dispatchEvent(new Event('blur'));");
}

bool ArenaWindow::eventFilter(QObject *object, QEvent *event) {
  if (event->type() == QEvent::ApplicationDeactivate || (object == this && event->type() == QEvent::WindowDeactivate)) neutral();
  return QMainWindow::eventFilter(object, event);
}

void ArenaWindow::stopHost() {
  if (m_host.state() == QProcess::NotRunning) return;
  m_stopping = true; m_host.write("shutdown\n"); m_host.closeWriteChannel();
  if (!m_host.waitForFinished(15000)) { m_host.terminate(); if (!m_host.waitForFinished(2000)) { m_host.kill(); m_host.waitForFinished(2000); } }
}

ArenaWindow::~ArenaWindow() { qApp->removeEventFilter(this); m_connection->leave(); stopHost(); disconnect(&m_host, nullptr, this, nullptr); }
void ArenaWindow::closeEvent(QCloseEvent *event) { neutral(); m_connection->leave(); stopHost(); event->accept(); }

} // namespace RM
