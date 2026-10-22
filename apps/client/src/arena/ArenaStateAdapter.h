#pragma once

#include <QObject>
#include <QJsonObject>
#include <QVariantMap>

namespace RM {

class ArenaStateAdapter : public QObject {
  Q_OBJECT
  Q_PROPERTY(QVariantMap player READ player NOTIFY changed)
  Q_PROPERTY(QVariantMap arena READ arena NOTIFY changed)
  Q_PROPERTY(bool online READ online NOTIFY changed)
public:
  explicit ArenaStateAdapter(QObject *parent = nullptr) : QObject(parent) {}
  QVariantMap player() const { return m_player; }
  QVariantMap arena() const { return m_arena; }
  bool online() const { return m_online; }
  void apply(const QJsonObject &snapshot) {
    const auto staticFrame = snapshot.value("static").toObject();
    if (!staticFrame.isEmpty()) m_arena = staticFrame.value("arena").toObject().toVariantMap();
    m_player = snapshot.value("player").toObject().toVariantMap();
    m_online = !m_player.isEmpty();
    emit changed();
  }
signals:
  void changed();
private:
  QVariantMap m_player, m_arena;
  bool m_online = false;
};

class ArenaBridge : public QObject {
  Q_OBJECT
public:
  using QObject::QObject;
public slots:
  void input(const QString &payload) { emit inputReceived(payload); }
  void viewportReady() { emit ready(); }
  void pause() { emit paused(); }
  void renderStats(const QString &payload) { emit renderStatsReceived(payload); }
signals:
  void snapshot(const QString &payload);
  void inputReceived(const QString &payload);
  void ready();
  void paused();
  void renderStatsReceived(const QString &payload);
};

} // namespace RM
