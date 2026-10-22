#pragma once

#include <QMainWindow>
#include <QJsonObject>
#include <QProcess>
#include <QMap>

class QLineEdit;
class QSpinBox;
class QComboBox;
class QLabel;
class QPushButton;
class QQuickWidget;
class QWebEngineView;
class QCloseEvent;

namespace RM {
class ArenaConnection;
class ArenaStateAdapter;
class ArenaBridge;

class ArenaWindow : public QMainWindow {
  Q_OBJECT
public:
  explicit ArenaWindow(QWidget *parent = nullptr);
  ~ArenaWindow() override;
protected:
  void closeEvent(QCloseEvent *event) override;
  bool eventFilter(QObject *object, QEvent *event) override;
private:
  void host();
  void join(const QString &key = {});
  void updateRoom(const QJsonObject &room);
  void onSnapshot(const QJsonObject &snapshot);
  void stopHost();
  void neutral();
  QString runtimeRoot() const;
  ArenaConnection *m_connection;
  ArenaStateAdapter *m_state;
  ArenaBridge *m_bridge;
  QWebEngineView *m_view;
  QQuickWidget *m_hud;
  QLineEdit *m_address, *m_nickname, *m_runtime, *m_ownerKey;
  QSpinBox *m_port;
  QComboBox *m_difficulty, *m_posture, *m_airTarget, *m_profile;
  QLabel *m_status, *m_roomLabel, *m_robotLabel;
  QPushButton *m_ready, *m_start;
  QMap<QString, QPushButton *> m_seats;
  QMap<QString, QPushButton *> m_actions;
  QProcess m_host;
  QByteArray m_hostOutput;
  QString m_playerId, m_loadedUrl, m_localAddresses;
  QJsonObject m_lastSnapshot, m_lastInput;
  bool m_isReady = false, m_stopping = false;
};

} // namespace RM
