#include "ArenaWindow.h"
#include "ArenaConnection.h"
#include "ArenaBridge.h"
#include "LocalPage.h"
#include <QApplication>
#include <QCloseEvent>
#include <QDebug>
#include <QDir>
#include <QFileDialog>
#include <QJsonDocument>
#include <QJsonArray>
#include <QNetworkInterface>
#include <QProcessEnvironment>
#include <QSettings>
#include <QStandardPaths>
#include <QWebChannel>
#include <QWebEngineView>
#include <QWebEnginePage>
#include <QWebEngineScript>
#include <QWebEngineScriptCollection>
#include <QWebEngineProfile>
#include <QTimer>
#include <QSet>

namespace RM {
class ArenaWebPage : public QWebEnginePage {
public: using QWebEnginePage::QWebEnginePage;
protected:
  void javaScriptConsoleMessage(JavaScriptConsoleMessageLevel level, const QString &message, int line, const QString &source) override {
    qWarning().noquote() << "ARENA_WEB" << int(level) << source << line << message;
  }
};
ArenaWindow::ArenaWindow(QWidget *parent) : QMainWindow(parent) {
  setWindowTitle(QStringLiteral("RMUC 2027 · ARENA")); resize(1600,900); setMinimumSize(1280,720);
  m_connection = new ArenaConnection(this); m_bridge = new ArenaBridge(this);
  QSettings settings;
  m_options = {{"nickname",settings.value("arena/nickname","玩家").toString()}, {"address",settings.value("arena/host","127.0.0.1").toString()}, {"port",settings.value("arena/port",8766).toInt()}, {"runtime",runtimeRoot()},{"sensitivity",settings.value("arena/sensitivity",20).toInt()},{"volume",settings.value("arena/volume",35).toInt()}};
  m_view = new QWebEngineView(this); setCentralWidget(m_view);
  auto *page = new ArenaWebPage(m_view); m_view->setPage(page); m_view->setContextMenuPolicy(Qt::NoContextMenu);
  m_webRoot = webRoot(); QFile versionFile(m_webRoot+"/version.json"); if(versionFile.open(QIODevice::ReadOnly)) m_connection->setResourceVersion(QJsonDocument::fromJson(versionFile.readAll()).object().value("resource_version").toString()); m_localPage = new LocalPage(m_webRoot,page);
  auto *channel = new QWebChannel(page); channel->registerObject("arena",m_bridge); page->setWebChannel(channel);
  QFile source(":/qtwebchannel/qwebchannel.js");
  if (source.open(QIODevice::ReadOnly)) {
    QWebEngineScript script; script.setName("arena-webchannel"); script.setInjectionPoint(QWebEngineScript::DocumentCreation); script.setWorldId(QWebEngineScript::MainWorld); script.setSourceCode(QString::fromUtf8(source.readAll())); page->scripts().insert(script);
  }
  connect(page,&QWebEnginePage::featurePermissionRequested,this,[this,page](const QUrl &origin,QWebEnginePage::Feature feature){ page->setFeaturePermission(origin,feature,origin.host()=="127.0.0.1" && origin.port()==m_localPage->serverPort() && feature==QWebEnginePage::MouseLock ? QWebEnginePage::PermissionGrantedByUser : QWebEnginePage::PermissionDeniedByUser); });
  connect(m_connection,&ArenaConnection::snapshotReceived,this,&ArenaWindow::onSnapshot);
  connect(m_connection,&ArenaConnection::roomChanged,this,[this](const QJsonObject &room){emit m_bridge->room(QString::fromUtf8(QJsonDocument(room).toJson(QJsonDocument::Compact)));});
  connect(m_connection,&ArenaConnection::catalogReceived,this,[this](const QJsonObject &catalog){emit m_bridge->response("catalog",QString::fromUtf8(QJsonDocument(catalog).toJson(QJsonDocument::Compact)));});
  connect(m_connection,&ArenaConnection::error,m_bridge,&ArenaBridge::notice);
  connect(m_connection,&ArenaConnection::connectionChanged,this,[this](bool connected){if(!connected){neutral();emit m_bridge->snapshot("{\"transport\":false}");}else m_connection->fetchCatalog();});
  connect(m_bridge,&ArenaBridge::ready,this,[this]{emit m_bridge->response("settings",QString::fromUtf8(QJsonDocument(m_options).toJson(QJsonDocument::Compact)));emit m_bridge->snapshot(QString::fromUtf8(QJsonDocument(m_lastSnapshot).toJson(QJsonDocument::Compact)));});
  connect(m_bridge,&ArenaBridge::inputReceived,this,[this](const QString &payload){m_lastInput=QJsonDocument::fromJson(payload.toUtf8()).object();if(!m_lastInput.isEmpty())m_connection->sendInput(m_lastInput);});
  connect(m_bridge,&ArenaBridge::paused,this,&ArenaWindow::neutral);
  connect(m_bridge,&ArenaBridge::renderStatsReceived,this,[](const QString &payload){qInfo().noquote()<<"ARENA_RENDER_TIMING"<<payload;});
  connect(m_bridge,&ArenaBridge::commandReceived,this,[this](const QString &payload){
    if(payload.size()>16384)return;
    const auto message=QJsonDocument::fromJson(payload.toUtf8()).object(); const QString name=message.value("name").toString(); const auto body=message.value("body").toObject();
    if(name=="host"||name=="join"){if(body.value("port").toInt()<1024||body.value("port").toInt()>65535||body.value("nickname").toString().trimmed().isEmpty()){emit m_bridge->notice(QStringLiteral("请填写昵称和 1024–65535 的端口"));return;}configure(body);if(name=="host")host();else join(body.value("owner_key").toString());}
    else if(name=="catalog")m_connection->fetchCatalog();
    else if(name=="seat"||name=="ready"||name=="start"||name=="profile"||name=="seat-config"||name=="team-config"||name=="stage-ready"||name=="skip-intro")m_connection->request(name,body);
    else if(name=="room")emit m_bridge->response("room","{}");
    else if(name=="settings"){configure(body);emit m_bridge->response("settings",QString::fromUtf8(QJsonDocument(m_options).toJson(QJsonDocument::Compact)));}
    else if(name=="leave"){neutral();m_connection->leave();stopHost();m_lastSnapshot={};emit m_bridge->snapshot("{\"player\":null,\"room\":null}");}
    else if(name=="browse-runtime") {const QString directory=QFileDialog::getExistingDirectory(this,QStringLiteral("选择仿真工程或运行包"),m_options.value("runtime").toString());if(!directory.isEmpty()){m_options.insert("runtime",directory);emit m_bridge->response("settings",QString::fromUtf8(QJsonDocument(m_options).toJson(QJsonDocument::Compact)));}}
    else if(name=="import-strategy"){
      const QString path=QFileDialog::getExistingDirectory(this,QStringLiteral("选择包含 manifest.json 的策略目录"));if(path.isEmpty())return;
      QJsonObject files;QDir dir(path);qint64 total=0;
      const QStringList allowed{"manifest.json","profile.json","showcase.json","strategy.py","robot_1.py","robot_3.py","robot_4.py","robot_6.py","robot_7.py"};
      for(const QString &entry:allowed){QFile file(dir.filePath(entry));if(!file.exists())continue;if(QFileInfo(file).isSymLink()||!file.open(QIODevice::ReadOnly)||file.size()>512*1024){emit m_bridge->notice(QStringLiteral("策略文件不可读取或超过大小限制"));return;}const auto data=file.readAll();total+=data.size();files.insert(entry,QString::fromUtf8(data));}
      if(total>4*1024*1024){emit m_bridge->notice(QStringLiteral("策略包过大"));return;}m_connection->request("strategy-import",{{"files",files}});
    }else if(name=="save") {neutral();const QString path=QFileDialog::getSaveFileName(this,QStringLiteral("保存本局回放"),"rmuc2027-replay.json","RMUC replay (*.json)");if(!path.isEmpty())m_connection->downloadReplay(path);}
    else {const QSet<QString> operations{"resume","pick","assemble","cancel","install","revive","heal","ammo","posture","air_target","air_launch","air_return"};if(operations.contains(name))m_connection->operation(name,message.value("value").toString());}
  });
  connect(&m_host,&QProcess::readyReadStandardOutput,this,[this]{m_hostOutput+=m_host.readAllStandardOutput();while(m_hostOutput.contains('\n')){const int end=m_hostOutput.indexOf('\n');const auto line=m_hostOutput.left(end);m_hostOutput.remove(0,end+1);const auto ready=QJsonDocument::fromJson(line).object();if(ready.value("lan_ready").toBool()){m_options.insert("address","127.0.0.1");join(ready.value("owner_key").toString());}}});
  connect(&m_host,&QProcess::readyReadStandardError,this,[this]{const QString message=QString::fromUtf8(m_host.readAllStandardError()).trimmed();if(!message.isEmpty())emit m_bridge->notice(message.right(800));});
  connect(&m_host,&QProcess::errorOccurred,this,[this](QProcess::ProcessError){emit m_bridge->notice(QStringLiteral("启动主机失败：")+m_host.errorString());});
  connect(&m_host,QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished),this,[this](int code,QProcess::ExitStatus){if(!m_stopping)emit m_bridge->notice(QStringLiteral("主机已退出：%1").arg(code));});
  if(m_webRoot.isEmpty())m_view->setHtml(QStringLiteral("<h2>缺少界面资源，请先执行 bun run build 并重新构建客户端。</h2>"));else if(!m_localPage->isListening())m_view->setHtml(QStringLiteral("<h2>无法启动本地界面服务，请检查回环网络权限。</h2>"));else m_view->load(m_localPage->pageUrl());
  qApp->installEventFilter(this);
}
void ArenaWindow::configure(const QJsonObject &body){for(const QString &key:{"nickname","address","port","runtime","sensitivity","volume"})if(body.contains(key))m_options.insert(key,body.value(key));QSettings s;s.setValue("arena/nickname",m_options.value("nickname").toString());s.setValue("arena/host",m_options.value("address").toString());s.setValue("arena/port",m_options.value("port").toInt());s.setValue("arena/runtime",m_options.value("runtime").toString());s.setValue("arena/sensitivity",m_options.value("sensitivity").toInt());s.setValue("arena/volume",m_options.value("volume").toInt());}
QString ArenaWindow::runtimeRoot() const {
  const QString env=qEnvironmentVariable("RMUC_ARENA_ROOT");if(!env.isEmpty())return env;
  const QString configured=QSettings().value("arena/runtime").toString();if(!configured.isEmpty())return configured;
  for(const QString &origin:{QCoreApplication::applicationDirPath(),QDir::currentPath()}){QDir parent(origin);for(int depth=0;depth<7;++depth){for(const QString &suffix:{QString(),QStringLiteral("/runtime/app")}){const QString path=parent.absolutePath()+suffix;if(QFileInfo::exists(path+"/apps/server/rmuc2027_server/server.py"))return path;}if(!parent.cdUp())break;}}return {};
}
QString ArenaWindow::webRoot() const {
  const QStringList candidates{QCoreApplication::applicationDirPath()+"/web",runtimeRoot()+"/build/web",QDir::currentPath()+"/build/web"};for(const QString &root:candidates)if(QFileInfo::exists(root+"/apps/client/web/index.html"))return root;return {};
}
void ArenaWindow::host(){
  if(m_host.state()!=QProcess::NotRunning||m_connection->hasSession()){emit m_bridge->notice(QStringLiteral("请先离开当前房间"));return;}
  QDir root(m_options.value("runtime").toString());if(QFileInfo::exists(root.filePath("app/apps/server/rmuc2027_server/server.py")))root.cd("app");
#ifdef Q_OS_WIN
  const QStringList interpreters{root.filePath("../python/python.exe"),root.filePath(".venv/Scripts/python.exe")};const QString engine=root.filePath("target/release/rmuc-sim.exe");
#else
  const QStringList interpreters{root.filePath("../python/bin/python3"),root.filePath(".venv/bin/python")};const QString engine=root.filePath("target/release/rmuc-sim");
#endif
  QString python=qEnvironmentVariable("RMUC_PYTHON");if(python.isEmpty())for(const QString &path:interpreters)if(QFileInfo::exists(path)){python=path;break;}
  if(python.isEmpty()||!QFileInfo::exists(engine)||!QFileInfo::exists(root.filePath("apps/server/rmuc2027_server/server.py"))){emit m_bridge->notice(QStringLiteral("仿真目录缺少 Python、服务或引擎，请先构建主机"));return;}
  m_host.setWorkingDirectory(root.absolutePath());m_hostOutput.clear();m_stopping=false;auto env=QProcessEnvironment::systemEnvironment();env.insert("PYTHONUTF8","1");env.insert("PYTHONDONTWRITEBYTECODE","1");env.insert("RMUC_RESOURCE_ROOT",root.absolutePath());env.insert("PYTHONPATH",root.filePath("packages/core")+QDir::listSeparator()+root.filePath("apps/server"));env.insert("RMUC_USER_DATA",QDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation)).filePath("arena"));m_host.setProcessEnvironment(env);m_host.start(python,{"-u","-m","rmuc2027_server","host","--port",QString::number(m_options.value("port").toInt(8766)),"--parent-stdin"});
}
void ArenaWindow::join(const QString &key){QUrl url;url.setScheme("http");url.setHost(m_options.value("address").toString());url.setPort(m_options.value("port").toInt(8766));m_lastSnapshot={};m_connection->join(url,m_options.value("nickname").toString(),key);}
void ArenaWindow::onSnapshot(const QJsonObject &snapshot){if(!snapshot.contains("room")&&snapshot.value("player").isNull()){emit m_bridge->snapshot("{\"transport\":false}");return;}const auto oldStatic=m_lastSnapshot.value("static"),oldMatch=m_lastSnapshot.value("room").toObject().value("match_id"),newMatch=snapshot.value("room").toObject().value("match_id");m_lastSnapshot=snapshot;if(!snapshot.contains("static")&&oldStatic.isObject()&&oldMatch==newMatch)m_lastSnapshot.insert("static",oldStatic);emit m_bridge->snapshot(QString::fromUtf8(QJsonDocument(snapshot).toJson(QJsonDocument::Compact)));}
void ArenaWindow::neutral(){m_connection->sendInput({{"yaw",m_lastInput.value("yaw").toDouble()},{"pitch",m_lastInput.value("pitch").toDouble()},{"active",false}});m_connection->operation("cancel");m_view->page()->runJavaScript("document.exitPointerLock();window.dispatchEvent(new Event('blur'));");}
bool ArenaWindow::eventFilter(QObject *object,QEvent *event){if(event->type()==QEvent::ApplicationDeactivate||(object==this&&event->type()==QEvent::WindowDeactivate))neutral();return QMainWindow::eventFilter(object,event);}
void ArenaWindow::stopHost(){if(m_host.state()==QProcess::NotRunning)return;m_stopping=true;m_host.write("shutdown\n");m_host.closeWriteChannel();if(!m_host.waitForFinished(15000)){m_host.terminate();if(!m_host.waitForFinished(2000)){m_host.kill();m_host.waitForFinished(2000);}}}
ArenaWindow::~ArenaWindow(){qApp->removeEventFilter(this);m_connection->leave();stopHost();disconnect(&m_host,nullptr,this,nullptr);}
void ArenaWindow::closeEvent(QCloseEvent *event){neutral();m_connection->leave();stopHost();event->accept();}
}
