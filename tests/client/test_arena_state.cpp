#include "arena/ArenaBridge.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QSignalSpy>
#include <QtTest/QtTest>
class TestArenaState : public QObject {
 Q_OBJECT
private slots:
 void forwardsStructuredOperations() {
  RM::ArenaBridge bridge; QSignalSpy spy(&bridge, &RM::ArenaBridge::commandReceived);
  bridge.command(QStringLiteral("{\"name\":\"assemble\",\"value\":\"3\"}"));
  QCOMPARE(spy.size(), 1);
  const auto command = QJsonDocument::fromJson(spy.at(0).at(0).toString().toUtf8()).object();
  QCOMPARE(command.value("name").toString(), QStringLiteral("assemble"));
  QCOMPARE(command.value("value").toString(), QStringLiteral("3"));
 }
 void viewportHandshakeAndNeutralInput() {
  RM::ArenaBridge bridge; QSignalSpy ready(&bridge, &RM::ArenaBridge::ready), inputs(&bridge, &RM::ArenaBridge::inputReceived);
  bridge.viewportReady(); bridge.input(QStringLiteral("{\"active\":false,\"fire\":false}"));
  QCOMPARE(ready.size(), 1); QCOMPARE(inputs.size(), 1);
  const auto input = QJsonDocument::fromJson(inputs.at(0).at(0).toString().toUtf8()).object();
  QVERIFY(!input.value("active").toBool()); QVERIFY(!input.value("fire").toBool());
 }
};
QTEST_GUILESS_MAIN(TestArenaState)
#include "test_arena_state.moc"
