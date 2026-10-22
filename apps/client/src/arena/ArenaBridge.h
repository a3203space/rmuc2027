#pragma once
#include <QObject>

namespace RM {
class ArenaBridge : public QObject {
  Q_OBJECT
public:
  using QObject::QObject;
public slots:
  void command(const QString &payload) { emit commandReceived(payload); }
  void input(const QString &payload) { emit inputReceived(payload); }
  void viewportReady() { emit ready(); }
  void pause() { emit paused(); }
  void renderStats(const QString &payload) { emit renderStatsReceived(payload); }
signals:
  void response(const QString &name, const QString &payload);
  void notice(const QString &message);
  void room(const QString &payload);
  void commandReceived(const QString &payload);
  void snapshot(const QString &payload);
  void inputReceived(const QString &payload);
  void ready();
  void paused();
  void renderStatsReceived(const QString &payload);
};

} // namespace RM
