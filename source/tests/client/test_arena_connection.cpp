#include "arena/ArenaConnection.h"
#include <QPointer>
#include <QSignalSpy>
#include <QTcpServer>
#include <QTcpSocket>
#include <QtTest/QtTest>

// Delayed real HTTP replies exercise cancellation between the health request,
// session allocation, and the WebSocket handshake.
class DelayedArenaHost : public QTcpServer {
public:
  int healthRequests = 0, joins = 0, leaves = 0;
  bool holdHealth = true, holdJoin = true, releasedToken = false;
  int profileStatus = 200, profileRequests = 0;
  QJsonObject lastProfile;
  QPointer<QTcpSocket> healthSocket, joinSocket;

  DelayedArenaHost() {
    connect(this, &QTcpServer::newConnection, this, [this] {
      while (hasPendingConnections()) {
        auto *socket = nextPendingConnection();
        connect(socket, &QTcpSocket::disconnected, socket, &QObject::deleteLater);
        connect(socket, &QTcpSocket::readyRead, this, [this, socket] {
          const QByteArray data = socket->property("request").toByteArray() + socket->readAll();
          socket->setProperty("request", data);
          if (!data.contains("\r\n\r\n") || socket->property("handled").toBool()) return;
          const int headerEnd = data.indexOf("\r\n\r\n");
          int contentLength = 0;
          for (const auto &line : data.left(headerEnd).split('\n'))
            if (line.trimmed().toLower().startsWith("content-length:")) contentLength=line.mid(line.indexOf(':')+1).trimmed().toInt();
          if (data.size() < headerEnd + 4 + contentLength) return;
          socket->setProperty("handled", true);
          const QByteArray path = data.split('\n').first().split(' ').value(1);
          if (path == "/api/lan/v2/health") {
            ++healthRequests; healthSocket = socket;
            if (!holdHealth) sendHealth();
          } else if (path == "/api/lan/v2/join") {
            ++joins; joinSocket = socket;
            if (!holdJoin) sendJoin();
          } else if (path == "/api/lan/v2/profile") {
            ++profileRequests;
            lastProfile = QJsonDocument::fromJson(data.mid(headerEnd+4,contentLength)).object();
            respond(socket, profileStatus==200?QByteArray(R"({"protocol":2,"revision":5,"status":"lobby","players":[]})"):QByteArray(R"({"detail":"profile locked"})"), profileStatus);
          } else if (path == "/api/lan/v2/leave") {
            ++leaves; releasedToken = data.contains("Bearer late-token");
            respond(socket, "{}");
          } else respond(socket, "{}", 400);
        });
      }
    });
  }
  QUrl url() const { return QUrl(QStringLiteral("http://127.0.0.1:%1").arg(serverPort())); }
  void sendHealth() { respond(healthSocket, R"({"protocol":2,"service":"arena-lan"})"); }
  void sendJoin() { respond(joinSocket, R"({"token":"late-token","room":{"protocol":2,"status":"lobby","players":[],"match_id":""}})"); }
  static void respond(QTcpSocket *socket, const QByteArray &body, int status = 200) {
    if (!socket) return;
    socket->write("HTTP/1.1 " + QByteArray::number(status) + " OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: " + QByteArray::number(body.size()) + "\r\n\r\n" + body);
    socket->disconnectFromHost();
  }
};

class TestArenaConnection : public QObject {
  Q_OBJECT
private slots:
  void profileSaveReportsSuccessWithRequestId() {
    DelayedArenaHost host; host.holdHealth = false; host.holdJoin = false;
    if (!host.listen(QHostAddress::LocalHost, 0)) QSKIP("TCP listen unavailable; profile HTTP check requires a network-capable environment");
    RM::ArenaConnection connection;
    QSignalSpy rooms(&connection, &RM::ArenaConnection::roomChanged), finished(&connection, &RM::ArenaConnection::profileFinished);
    QVERIFY(connection.join(host.url(), QStringLiteral("玩家")));
    QTRY_COMPARE(rooms.count(), 1);
    connection.request(QStringLiteral("profile"), {{"chassis","hp"},{"weapon","cooling"}}, QStringLiteral("save-hp-cooling"));
    QTRY_COMPARE(finished.count(), 1);
    QCOMPARE(host.profileRequests, 1);
    QCOMPARE(host.lastProfile, (QJsonObject{{"chassis","hp"},{"weapon","cooling"}}));
    const auto result=finished.at(0).at(0).toJsonObject();
    QVERIFY(result.value("ok").toBool());
    QCOMPARE(result.value("request_id").toString(), QStringLiteral("save-hp-cooling"));
    QCOMPARE(result.value("room").toObject().value("revision").toInt(), 5);
    connection.leave();
  }
  void profileSaveReportsRejectionWithoutChangingRoom() {
    DelayedArenaHost host; host.holdHealth = false; host.holdJoin = false; host.profileStatus = 400;
    if (!host.listen(QHostAddress::LocalHost, 0)) QSKIP("TCP listen unavailable; profile HTTP check requires a network-capable environment");
    RM::ArenaConnection connection;
    QSignalSpy rooms(&connection, &RM::ArenaConnection::roomChanged), finished(&connection, &RM::ArenaConnection::profileFinished);
    QVERIFY(connection.join(host.url(), QStringLiteral("玩家")));
    QTRY_COMPARE(rooms.count(), 1);
    connection.request(QStringLiteral("profile"), {{"chassis","power"},{"weapon","burst"}}, QStringLiteral("save-rejected"));
    QTRY_COMPARE(finished.count(), 1);
    const auto result=finished.at(0).at(0).toJsonObject();
    QVERIFY(!result.value("ok").toBool());
    QCOMPARE(result.value("request_id").toString(), QStringLiteral("save-rejected"));
    QCOMPARE(result.value("message").toString(), QStringLiteral("profile locked"));
    QCOMPARE(rooms.count(), 1);
    connection.leave();
  }
  void cancelDuringHealthDoesNotCreateSession() {
    DelayedArenaHost host;
    QVERIFY(host.listen(QHostAddress::LocalHost, 0));
    RM::ArenaConnection connection;
    QSignalSpy rooms(&connection, &RM::ArenaConnection::roomChanged);
    connection.join(host.url(), QStringLiteral("玩家"));
    QVERIFY(connection.hasSession());
    QTRY_COMPARE(host.healthRequests, 1);
    connection.leave();
    QVERIFY(!connection.hasSession());
    host.sendHealth();
    QTest::qWait(200);
    QCOMPARE(host.joins, 0);
    QCOMPARE(rooms.count(), 0);
  }
  void duplicateClickCreatesOnlyOneSession() {
    DelayedArenaHost host; host.holdJoin = false;
    QVERIFY(host.listen(QHostAddress::LocalHost, 0));
    RM::ArenaConnection connection;
    QSignalSpy rooms(&connection, &RM::ArenaConnection::roomChanged);
    QVERIFY(connection.join(host.url(), QStringLiteral("玩家")));
    QVERIFY(!connection.join(host.url(), QStringLiteral("玩家")));
    QTRY_COMPARE(host.healthRequests, 1);
    host.sendHealth();
    QTRY_COMPARE(rooms.count(), 1);
    QVERIFY(connection.hasSession());
    QCOMPARE(host.joins, 1);
  }
  void cancelledAllocationIsReleasedWithoutReconnecting() {
    DelayedArenaHost host; host.holdHealth = false;
    QVERIFY(host.listen(QHostAddress::LocalHost, 0));
    RM::ArenaConnection connection;
    QSignalSpy rooms(&connection, &RM::ArenaConnection::roomChanged);
    connection.join(host.url(), QStringLiteral("玩家"));
    QTRY_COMPARE(host.joins, 1);
    connection.leave(); host.sendJoin();
    QTRY_COMPARE(host.leaves, 1);
    QVERIFY(host.releasedToken);
    QCOMPARE(rooms.count(), 0);
    QVERIFY(!connection.isConnected());
  }
};

QTEST_GUILESS_MAIN(TestArenaConnection)
#include "test_arena_connection.moc"
