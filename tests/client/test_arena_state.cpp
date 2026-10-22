#include "arena/ArenaStateAdapter.h"
#include <QQuickWidget>
#include <QQmlContext>
#include <QQuickItem>
#include <QSignalSpy>
#include <QtTest/QtTest>
class TestArenaState : public QObject {
 Q_OBJECT
private slots:
 void initTestCase() { Q_INIT_RESOURCE(qml); }
 void hudLoads() {
  RM::ArenaStateAdapter adapter; QQuickWidget view;
  view.rootContext()->setContextProperty("arenaState", &adapter);
  view.setSource(QUrl("qrc:/qml/ArenaHud.qml"));
  QTRY_COMPARE(view.status(), QQuickWidget::Ready);
  QVERIFY(view.rootObject());
 }
 void stateComesOnlyFromHost() {
  RM::ArenaStateAdapter adapter;
  adapter.apply({{"player", QJsonObject{{"slot",3},{"team","blue"},{"self",QJsonObject{{"hp",350}}},{"scene",QJsonObject{{"remaining",307.4}}}}}});
  QVERIFY(adapter.online());
  QCOMPARE(adapter.player().value("slot").toInt(),3);
  QTest::qWait(1100);
  QCOMPARE(adapter.player().value("scene").toMap().value("remaining").toDouble(),307.4);
  adapter.apply({}); QVERIFY(!adapter.online());
 }
 void penaltyContainsNoFakeTacticalState() {
  RM::ArenaStateAdapter adapter;
  adapter.apply({{"player",QJsonObject{{"display",QJsonObject{{"ui_blocked",true}}}}}});
  QVERIFY(adapter.player().value("display").toMap().value("ui_blocked").toBool());
  QVERIFY(!adapter.player().contains("tactical"));
 }
};
QTEST_MAIN(TestArenaState)
#include "test_arena_state.moc"
