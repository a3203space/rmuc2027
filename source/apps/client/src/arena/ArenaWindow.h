#pragma once
#include <QMainWindow>
#include <QJsonObject>
#include <QProcess>
class QWebEngineView;
class QCloseEvent;
namespace RM {
class ArenaConnection;
class ArenaBridge;
class LocalPage;
class ArenaWindow : public QMainWindow {
  Q_OBJECT
public:
  explicit ArenaWindow(QWidget *parent = nullptr);
  ~ArenaWindow() override;
protected:
  bool eventFilter(QObject *, QEvent *) override;
  void closeEvent(QCloseEvent *) override;
private:
  void host();
  void join(const QString &key = {});
  void onSnapshot(const QJsonObject &);
  void neutral();
  void stopHost();
  void configure(const QJsonObject &);
  QString runtimeRoot() const;
  QString webRoot() const;
  ArenaConnection *m_connection;
  ArenaBridge *m_bridge;
  QWebEngineView *m_view;
  LocalPage *m_localPage;
  QProcess m_host;
  QByteArray m_hostOutput;
  QJsonObject m_lastSnapshot, m_lastInput, m_options;
  QString m_webRoot;
  bool m_stopping = false;
};
}
