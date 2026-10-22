#pragma once
#include <QTcpServer>
#include <QTcpSocket>
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QMimeDatabase>
#include <QUrl>
#include <QVariant>
namespace RM {
// Serve the installed frontend over loopback. Standard HTTP keeps fetch(),
// glTF streams and seekable audio compatible with Qt WebEngine 6.2.
class LocalPage : public QTcpServer {
public:
  explicit LocalPage(const QString &root, QObject *parent) : QTcpServer(parent), m_root(QDir(root).canonicalPath()) {
    connect(this,&QTcpServer::newConnection,this,[this]{
      while(hasPendingConnections()) {
        auto *socket=nextPendingConnection();socket->setProperty("request",QByteArray());
        connect(socket,&QTcpSocket::disconnected,socket,&QObject::deleteLater);
        connect(socket,&QTcpSocket::readyRead,socket,[this,socket]{
          auto request=socket->property("request").toByteArray()+socket->readAll();
          if(request.size()>16384){socket->disconnectFromHost();return;}
          if(!request.contains("\r\n\r\n")){socket->setProperty("request",request);return;}
          if(socket->property("answered").toBool())return;socket->setProperty("answered",true);
          const auto lines=request.split('\n'),first=lines.value(0).trimmed().split(' ');
          if(first.size()<2){respond(socket,400,"Bad Request");return;}
          const bool head=first[0]=="HEAD";
          if(first[0]!="GET"&&!head){respond(socket,405,"Method Not Allowed");return;}
          const QString path=QUrl::fromEncoded(first[1]).path();
          const auto filePath=QFileInfo(m_root+(path=="/play"?"/apps/client/web/index.html":path)).canonicalFilePath();
          if(m_root.isEmpty()||!filePath.startsWith(m_root+"/")){respond(socket,404,"Not Found");return;}
          QFile file(filePath);if(!file.open(QIODevice::ReadOnly)){respond(socket,404,"Not Found");return;}
          qint64 start=0,end=file.size()-1;bool partial=false;
          for(const auto &line:lines)if(line.trimmed().toLower().startsWith("range:")){
            const auto range=line.trimmed().mid(6).trimmed();
            if(!range.startsWith("bytes=")||range.contains(',')){respond(socket,416,"Range Not Satisfiable");return;}
            const auto parts=range.mid(6).split('-');bool valid=false;
            if(parts.size()!=2){respond(socket,416,"Range Not Satisfiable");return;}
            if(parts[0].isEmpty()){const auto suffix=parts[1].toLongLong(&valid);if(!valid||suffix<=0){respond(socket,416,"Range Not Satisfiable");return;}start=qMax(qint64(0),file.size()-suffix);}
            else {start=parts[0].toLongLong(&valid);if(!valid){respond(socket,416,"Range Not Satisfiable");return;}if(!parts[1].isEmpty()){end=parts[1].toLongLong(&valid);if(!valid){respond(socket,416,"Range Not Satisfiable");return;}}}
            end=qMin(end,file.size()-1);if(start<0||start>end){respond(socket,416,"Range Not Satisfiable");return;}partial=true;
          }
          auto mime=QMimeDatabase().mimeTypeForFile(filePath).name().toUtf8();if(filePath.endsWith(".js"))mime="application/javascript";if(filePath.endsWith(".m4a"))mime="audio/mp4";
          const auto length=qMax(qint64(0),end-start+1);
          QByteArray headers=partial?"HTTP/1.1 206 Partial Content\r\n":"HTTP/1.1 200 OK\r\n";
          headers+="Content-Type: "+mime+"\r\nContent-Length: "+QByteArray::number(length)+"\r\nAccept-Ranges: bytes\r\nCache-Control: no-cache\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n";
          if(partial)headers+="Content-Range: bytes "+QByteArray::number(start)+"-"+QByteArray::number(end)+"/"+QByteArray::number(file.size())+"\r\n";
          socket->write(headers+"\r\n");if(!head){file.seek(start);socket->write(file.read(length));}socket->disconnectFromHost();
        });
      }
    });
    if(!m_root.isEmpty())listen(QHostAddress::LocalHost,0);
  }
  QUrl pageUrl() const { return QUrl(QString("http://127.0.0.1:%1/play").arg(serverPort())); }
private:
  static void respond(QTcpSocket *socket,int status,const QByteArray &message){socket->write("HTTP/1.1 "+QByteArray::number(status)+" "+message+"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");socket->disconnectFromHost();}
  QString m_root;
};
}
