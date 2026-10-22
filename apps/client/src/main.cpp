#include "arena/ArenaWindow.h"
#include <QApplication>
#include <QQuickWindow>
#include <QSGRendererInterface>
#include <QSurfaceFormat>
int main(int argc, char **argv) {
  QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
  // WebEngine 6.2 disables its GPU path with the software scene graph, which
  // prevents the Three.js viewport from creating the required WebGL 2 context.
  QQuickWindow::setGraphicsApi(QSGRendererInterface::OpenGL);
  QSurfaceFormat format; format.setAlphaBufferSize(8); QSurfaceFormat::setDefaultFormat(format);
  QApplication app(argc, argv);
  app.setApplicationName("RMUC2027Client");
  app.setOrganizationName("RMUC2027");
  app.setApplicationVersion("0.1.0");
  app.setStyle("Fusion");
  RM::ArenaWindow window; window.showFullScreen();
  return app.exec();
}
