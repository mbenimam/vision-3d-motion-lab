#include "app/theme/theme.hpp"

#include <QApplication>
#include <QFile>
#include <QPalette>
#include <QStyleFactory>

// The stylesheet lives in a static library, so its resources must be registered by hand.
static void initThemeResources() { Q_INIT_RESOURCE(theme); }

namespace vl::app {

void applyTheme(QApplication& app) {
    initThemeResources();
    app.setStyle(QStyleFactory::create(QStringLiteral("Fusion")));
    QPalette p;
    const QColor bg(0x0e, 0x11, 0x17), panel(0x12, 0x17, 0x22), text(0xd6, 0xdc, 0xe4), accent(0xff, 0xb0, 0x00);
    p.setColor(QPalette::Window, panel);
    p.setColor(QPalette::WindowText, text);
    p.setColor(QPalette::Base, bg);
    p.setColor(QPalette::AlternateBase, panel);
    p.setColor(QPalette::Text, text);
    p.setColor(QPalette::Button, panel);
    p.setColor(QPalette::ButtonText, text);
    p.setColor(QPalette::Highlight, accent);
    p.setColor(QPalette::HighlightedText, bg);
    p.setColor(QPalette::ToolTipBase, panel);
    p.setColor(QPalette::ToolTipText, text);
    p.setColor(QPalette::PlaceholderText, QColor(0x6b, 0x76, 0x86));
    p.setColor(QPalette::Disabled, QPalette::Text, QColor(0x5a, 0x63, 0x70));
    p.setColor(QPalette::Disabled, QPalette::ButtonText, QColor(0x5a, 0x63, 0x70));
    app.setPalette(p);
    QFile qss(QStringLiteral(":/theme/dark.qss"));
    if (qss.open(QIODevice::ReadOnly)) app.setStyleSheet(QString::fromUtf8(qss.readAll()));
}

}  // namespace vl::app
