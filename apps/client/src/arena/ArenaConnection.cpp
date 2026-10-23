#include "ArenaConnection.h"

#include <QDateTime>
#include <QDebug>
#include <QFile>
#include <QJsonDocument>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QSaveFile>

namespace RM {

ArenaConnection::ArenaConnection(QObject *parent) : QObject(parent) {
  m_clock.start();
  m_heartbeat.setInterval(1000);
  m_watchdog.setInterval(500);
  m_reconnect.setSingleShot(true);
  connect(&m_heartbeat, &QTimer::timeout, this, [this] { send({{"type", "ping"}}); });
  connect(&m_reconnect, &QTimer::timeout, this, &ArenaConnection::openSocket);
  connect(&m_watchdog, &QTimer::timeout, this, [this] {
    if (isConnected() && m_clock.elapsed() - m_lastSnapshot > 2500) {
      emit error(QStringLiteral("仿真状态已超时，正在重连"));
      m_socket.abort();
    }
  });
  connect(&m_socket, &QWebSocket::connected, this, [this] {
    m_lastSnapshot = m_clock.elapsed();
    send({{"type", "auth"}, {"protocol", 2}, {"token", m_token}});
    m_heartbeat.start(); m_watchdog.start();
    emit connectionChanged(true);
  });
  connect(&m_socket, &QWebSocket::disconnected, this, [this] {
    m_running = false;
    m_heartbeat.stop(); m_watchdog.stop();
    emit connectionChanged(false);
    emit snapshotReceived({{"type", "snapshot"}, {"player", QJsonValue::Null}});
    if (!m_leaving && !m_reconnect.isActive()) {
      m_reconnect.start(m_retryMs);
      m_retryMs = qMin(10000, m_retryMs * 2);
    }
  });
  connect(&m_socket, &QWebSocket::textMessageReceived, this, [this](const QString &text) {
    QJsonParseError parse;
    const QJsonDocument document = QJsonDocument::fromJson(text.toUtf8(), &parse);
    if (parse.error != QJsonParseError::NoError || !document.isObject()) {
      emit error(QStringLiteral("仿真主机返回了无效状态")); return;
    }
    const QJsonObject snapshot = document.object(), room = snapshot.value("room").toObject();
    if (snapshot.value("type").toString() == "error") {
      emit error(snapshot.value("message").toString());
      if (snapshot.value("fatal").toBool()) { m_leaving = true; m_socket.close(); }
      return;
    }
    m_lastSnapshot = m_clock.elapsed(); m_retryMs = 1000;
    if (!room.isEmpty()) {
      m_matchId = room.value("match_id").toString();
      m_running = room.value("status").toString() == "running";
      if (room.value("protocol").toInt() != 2) {
        emit error(QStringLiteral("联机协议版本不兼容")); leave(); return;
      }
      emit roomChanged(room);
      if (room.value("status").toString() == "closed") {
        emit error(QStringLiteral("房间已关闭"));
        m_leaving = true;
        m_socket.close();
      }
    }
    const qint64 ack = snapshot.value("ack_seq").toVariant().toLongLong();
    if (m_pendingInputs.contains(ack) && m_clock.elapsed() - m_lastTimingLog >= 1000) {
      qInfo() << "ARENA_INPUT_TIMING ack_seq=" << ack
              << "execution_roundtrip_ms=" << m_clock.elapsed() - m_pendingInputs.value(ack);
      m_lastTimingLog = m_clock.elapsed();
    }
    while (!m_pendingInputs.isEmpty() && m_pendingInputs.firstKey() <= ack) m_pendingInputs.erase(m_pendingInputs.begin());
    m_sequence = qMax(m_sequence, ack + 1);
    emit snapshotReceived(snapshot);
  });
#if QT_VERSION >= QT_VERSION_CHECK(6, 5, 0)
  connect(&m_socket, &QWebSocket::errorOccurred, this, [this](QAbstractSocket::SocketError) {
#else
  connect(&m_socket, QOverload<QAbstractSocket::SocketError>::of(&QWebSocket::error), this, [this](QAbstractSocket::SocketError) {
#endif
    emit error(QStringLiteral("联机连接失败：%1").arg(m_socket.errorString()));
    if (!m_leaving && !m_reconnect.isActive()) m_reconnect.start(m_retryMs);
  });
}

ArenaConnection::~ArenaConnection() {
  m_leaving = true;
  m_heartbeat.stop(); m_watchdog.stop(); m_reconnect.stop();
  disconnect(&m_socket, nullptr, this, nullptr);
  m_socket.abort();
  for (auto *reply : m_http.findChildren<QNetworkReply *>()) {
    disconnect(reply, nullptr, this, nullptr);
    reply->abort();
  }
}

bool ArenaConnection::join(const QUrl &url, const QString &nickname, const QString &hostKey) {
  if (m_joining) { emit error(QStringLiteral("正在连接房间，请等待连接结果")); return false; }
  if (!m_token.isEmpty() && !m_leaving) { emit error(QStringLiteral("请先离开当前房间")); return false; }
  if (url.scheme() != "http" || url.host().isEmpty() || !url.userInfo().isEmpty()) {
    emit error(QStringLiteral("请输入有效的局域网 HTTP 地址")); return false;
  }
  leave(); m_url = url; m_url.setPath(QString()); m_url.setQuery(QString()); m_url.setFragment(QString());
  m_joining = true;
  const quint64 epoch = m_epoch;
  m_token.clear(); m_matchId.clear(); m_sequence = 0; m_retryMs = 1000;
  QNetworkRequest request(m_url.resolved(QUrl("/api/lan/v2/health")));
  request.setTransferTimeout(5000);
  auto *reply = m_http.get(request);
  connect(reply, &QNetworkReply::finished, this, [this, reply, nickname, hostKey, epoch] {
    const QJsonObject health = QJsonDocument::fromJson(reply->readAll()).object();
    const bool ok = reply->error() == QNetworkReply::NoError && health.value("protocol").toInt() == 2 && health.value("service").toString() == "arena-lan";
    reply->deleteLater();
    if (epoch != m_epoch) return;
    if (!ok) { m_joining = false; emit error(QStringLiteral("无法连接 RMUC 2027 仿真主机，或协议版本不兼容")); return; }
    if (!m_resourceVersion.isEmpty() && health.value("resource_version").toString() != m_resourceVersion) {
      m_joining = false; emit error(QStringLiteral("客户端和主机界面资源版本不一致，请同步构建或更新运行包后重启主机")); return;
    }
    QJsonObject body{{"protocol", 2}, {"nickname", nickname}};
    if (!hostKey.isEmpty()) body.insert("host_key", hostKey);
    post("join", body, [this](const QJsonObject &value) {
      m_token = value.value("token").toString();
      if (m_token.isEmpty()) { emit error(QStringLiteral("主机没有分配会话")); return; }
      m_leaving = false;
      emit roomChanged(value.value("room").toObject());
      openSocket();
    });
  });
  return true;
}

void ArenaConnection::post(const QString &action, const QJsonObject &body,
                           std::function<void(const QJsonObject &)> callback, const QString &requestId) {
  const quint64 epoch = m_epoch;
  QNetworkRequest request(m_url.resolved(QUrl("/api/lan/v2/" + action)));
  request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
  request.setTransferTimeout(5000);
  if (!m_token.isEmpty()) request.setRawHeader("Authorization", "Bearer " + m_token.toUtf8());
  auto *reply = m_http.post(request, QJsonDocument(body).toJson(QJsonDocument::Compact));
  connect(reply, &QNetworkReply::finished, this, [this, reply, callback, action, epoch, requestId] {
    const QJsonObject value = QJsonDocument::fromJson(reply->readAll()).object();
    if (epoch != m_epoch) {
      // A cancelled join may already have allocated a server-side seat. Release
      // that session at its original host without adopting its token locally.
      const QString token = value.value("token").toString();
      if (action == "join" && !token.isEmpty()) releaseSession(reply->url(), token);
      reply->deleteLater(); return;
    }
    if (action == "join") m_joining = false;
    if (reply->error() != QNetworkReply::NoError) {
      const QString detail = value.value("detail").isString() ? value.value("detail").toString() : reply->errorString();
      emit error(detail);
      if (action == "profile") emit profileFinished({{"request_id",requestId},{"ok",false},{"message",detail}});
    } else if (callback) callback(value);
    else if (action == "strategy-import") emit catalogReceived(value);
    else emit roomChanged(value);
    if (action == "profile" && reply->error() == QNetworkReply::NoError)
      emit profileFinished({{"request_id",requestId},{"ok",true},{"room",value}});
    reply->deleteLater();
  });
}

void ArenaConnection::releaseSession(const QUrl &url, const QString &token) {
  QNetworkRequest request(url.resolved(QUrl("/api/lan/v2/leave")));
  request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
  request.setRawHeader("Authorization", "Bearer " + token.toUtf8());
  request.setTransferTimeout(5000);
  auto *reply = m_http.post(request, "{}");
  connect(reply, &QNetworkReply::finished, reply, &QObject::deleteLater);
}

void ArenaConnection::request(const QString &action, const QJsonObject &body, const QString &requestId) {
  if (m_token.isEmpty()) {
    emit error(QStringLiteral("请先加入房间"));
    if (action == "profile") emit profileFinished({{"request_id",requestId},{"ok",false},{"message",QStringLiteral("请先加入房间")}});
    return;
  }
  post(action, body, {}, requestId);
}

void ArenaConnection::fetchCatalog() {
  if (!m_token.isEmpty()) post("catalog", {}, [this](const QJsonObject &value) { emit catalogReceived(value); });
}

void ArenaConnection::openSocket() {
  if (m_leaving || m_token.isEmpty() || isConnected()) return;
  QUrl url = m_url.resolved(QUrl("/api/lan/v2/session")); url.setScheme("ws");
  m_socket.open(url);
}

void ArenaConnection::send(QJsonObject message) {
  if (isConnected()) m_socket.sendTextMessage(QString::fromUtf8(QJsonDocument(message).toJson(QJsonDocument::Compact)));
}

void ArenaConnection::sendInput(QJsonObject input) {
  if (m_matchId.isEmpty() || !m_running) return;
  input.insert("type", "input"); input.insert("match_id", m_matchId); input.insert("seq", m_sequence++);
  m_pendingInputs.insert(m_sequence - 1, m_clock.elapsed());
  while (m_pendingInputs.size() > 1000) m_pendingInputs.erase(m_pendingInputs.begin());
  send(input);
}

void ArenaConnection::operation(const QString &name, const QString &value) {
  if (m_matchId.isEmpty() || !m_running) return;
  send({{"type", "operation"}, {"match_id", m_matchId}, {"seq", m_sequence++}, {"name", name}, {"value", value}});
}

void ArenaConnection::leave() {
  ++m_epoch;
  m_joining = false;
  m_leaving = true; m_reconnect.stop(); m_heartbeat.stop(); m_watchdog.stop();
  m_running = false;
  m_pendingInputs.clear();
  if (!m_token.isEmpty()) post("leave", {}, [](const QJsonObject &) {});
  m_socket.close(); m_token.clear(); m_matchId.clear();
}

void ArenaConnection::downloadReplay(const QString &path) {
  auto *file = new QSaveFile(path, this);
  if (!file->open(QIODevice::WriteOnly)) {
    emit error(QStringLiteral("无法打开回放保存路径")); file->deleteLater(); return;
  }
  QNetworkRequest request(m_url.resolved(QUrl("/api/lan/v2/replay")));
  request.setRawHeader("Authorization", "Bearer " + m_token.toUtf8());
  auto *reply = m_http.get(request);
  connect(reply, &QNetworkReply::readyRead, this, [reply, file] {
    const QByteArray bytes = reply->readAll();
    if (file->write(bytes) != bytes.size()) reply->abort();
  });
  connect(reply, &QNetworkReply::finished, this, [this, reply, file] {
    if (reply->error() != QNetworkReply::NoError || !file->commit()) {
      file->cancelWriting();
      emit error(QStringLiteral("保存回放失败"));
    }
    file->deleteLater(); reply->deleteLater();
  });
}

} // namespace RM
