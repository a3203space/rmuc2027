#pragma once

#include <QObject>
#include <QJsonObject>
#include <QElapsedTimer>
#include <QMap>
#include <QNetworkAccessManager>
#include <QTimer>
#include <QUrl>
#include <QWebSocket>
#include <functional>

namespace RM {

class ArenaConnection : public QObject {
  Q_OBJECT
public:
  explicit ArenaConnection(QObject *parent = nullptr);
  ~ArenaConnection() override;
  bool join(const QUrl &url, const QString &nickname, const QString &hostKey = {});
  void request(const QString &action, const QJsonObject &body = {}, const QString &requestId = {});
  void fetchCatalog();
  void sendInput(QJsonObject input);
  void operation(const QString &name, const QString &value = {});
  void leave();
  void downloadReplay(const QString &path);
  QUrl serverUrl() const { return m_url; }
  bool isConnected() const { return m_socket.state() == QAbstractSocket::ConnectedState; }
  bool hasSession() const { return m_joining || (!m_token.isEmpty() && !m_leaving); }
  void setResourceVersion(const QString &value) { m_resourceVersion = value; }
signals:
  void catalogReceived(const QJsonObject &catalog);
  void snapshotReceived(const QJsonObject &snapshot);
  void roomChanged(const QJsonObject &room);
  void error(const QString &message);
  void connectionChanged(bool connected);
  void profileFinished(const QJsonObject &result);
private:
  void post(const QString &action, const QJsonObject &body,
            std::function<void(const QJsonObject &)> callback = {}, const QString &requestId = {});
  void openSocket();
  void releaseSession(const QUrl &url, const QString &token);
  void send(QJsonObject message);
  QNetworkAccessManager m_http;
  QWebSocket m_socket;
  QTimer m_heartbeat, m_reconnect, m_watchdog;
  QUrl m_url;
  QString m_token, m_matchId;
  QString m_resourceVersion;
  qint64 m_sequence = 0, m_lastSnapshot = 0;
  QElapsedTimer m_clock;
  QMap<qint64, qint64> m_pendingInputs;
  qint64 m_lastTimingLog = 0;
  int m_retryMs = 1000;
  bool m_leaving = true;
  bool m_running = false;
  quint64 m_epoch = 0;
  bool m_joining = false;
};

} // namespace RM
