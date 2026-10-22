#include "arena/ArenaWindow.h"
#include <QApplication>
#include <QQuickStyle>
#include <QQuickWindow>
#include <QSGRendererInterface>
#include <QSurfaceFormat>
int main(int argc, char **argv) {
  QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
  QQuickWindow::setGraphicsApi(qEnvironmentVariable("RM_QT_QUICK_BACKEND") == "opengl" ? QSGRendererInterface::OpenGL : QSGRendererInterface::Software);
  QSurfaceFormat format; format.setAlphaBufferSize(8); QSurfaceFormat::setDefaultFormat(format);
  QQuickStyle::setStyle("Fusion");
  Q_INIT_RESOURCE(qml);
  QApplication app(argc, argv);
  app.setApplicationName("RMUC2027Client");
  app.setOrganizationName("RMUC2027");
  app.setApplicationVersion("0.1.0");
  app.setStyle("Fusion");
  RM::ArenaWindow window; window.showFullScreen();
  return app.exec();
}
