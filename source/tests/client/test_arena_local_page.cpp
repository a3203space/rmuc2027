#include "arena/LocalPage.h"
#include <QTemporaryDir>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QSignalSpy>
#include <QVariant>
#include <QtTest/QtTest>
class TestLocalPage : public QObject {
 Q_OBJECT
private slots:
 void servesInstalledPageAndScripts() {
  QTemporaryDir temp;QDir().mkpath(temp.path()+"/apps/client/web");QDir().mkpath(temp.path()+"/assets");
  QFile index(temp.path()+"/apps/client/web/index.html");QVERIFY(index.open(QIODevice::WriteOnly));index.write("<h1>Arena offline home</h1>");index.close();
  QFile script(temp.path()+"/assets/app.js");QVERIFY(script.open(QIODevice::WriteOnly));script.write("window.arena=true;");script.close();
  RM::LocalPage server(temp.path(),this);if(!server.isListening())QSKIP("Environment forbids loopback TCP listening; desktop HTTP check pending");
  QNetworkAccessManager network;
  auto *page=network.get(QNetworkRequest(server.pageUrl()));QSignalSpy pageDone(page,&QNetworkReply::finished);QVERIFY(pageDone.wait(3000));QCOMPARE(page->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt(),200);QVERIFY(page->readAll().contains("Arena offline home"));page->deleteLater();
  auto *asset=network.get(QNetworkRequest(server.pageUrl().resolved(QUrl("/assets/app.js"))));QSignalSpy assetDone(asset,&QNetworkReply::finished);QVERIFY(assetDone.wait(3000));QCOMPARE(asset->header(QNetworkRequest::ContentTypeHeader).toString(),QString("application/javascript"));QCOMPARE(asset->readAll(),QByteArray("window.arena=true;"));asset->deleteLater();
 }
 void audioByteRangesAndMissingResources() {
  QTemporaryDir temp;QDir().mkpath(temp.path()+"/assets");QFile audio(temp.path()+"/assets/test.m4a");QVERIFY(audio.open(QIODevice::WriteOnly));audio.write("0123456789");audio.close();
  RM::LocalPage server(temp.path(),this);if(!server.isListening())QSKIP("Environment forbids loopback TCP listening; audio range check pending");
  QNetworkAccessManager network;QNetworkRequest request(server.pageUrl().resolved(QUrl("/assets/test.m4a")));request.setRawHeader("Range","bytes=3-6");
  auto *reply=network.get(request);QSignalSpy done(reply,&QNetworkReply::finished);QVERIFY(done.wait(3000));QCOMPARE(reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt(),206);QCOMPARE(reply->rawHeader("Content-Range"),QByteArray("bytes 3-6/10"));QCOMPARE(reply->readAll(),QByteArray("3456"));reply->deleteLater();
  auto *missing=network.get(QNetworkRequest(server.pageUrl().resolved(QUrl("/not-present"))));QSignalSpy missingDone(missing,&QNetworkReply::finished);QVERIFY(missingDone.wait(3000));QCOMPARE(missing->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt(),404);missing->deleteLater();
 }
};
QTEST_GUILESS_MAIN(TestLocalPage)
#include "test_arena_local_page.moc"
