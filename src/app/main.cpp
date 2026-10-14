// Vision Lab: an interactive lab for 3D geometry and motion from images.
//
//   vision-lab                         open the app on the last topic
//   vision-lab --module stereo-local --sample 1 --set radius=5 --view 3 --frame 10
//              --click 120,80 --screenshot out.png --size 1440x900
//   vision-lab --list                  print the topic ids
//
// The screenshot options render the window after the first complete result and quit; with
// QT_QPA_PLATFORM=offscreen they run headless (CI and README pictures).

#include <QApplication>
#include <QCommandLineParser>
#include <QScreen>
#include <QSettings>
#include <QTemporaryDir>
#include <QTimer>
#include <cstdio>

#include "app/controller/app_controller.hpp"
#include "app/theme/theme.hpp"
#include "app/view/main_window.hpp"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    QApplication::setOrganizationName(QStringLiteral("mbenimam"));
    QApplication::setApplicationName(QStringLiteral("vision-lab"));
    QApplication::setApplicationDisplayName(QStringLiteral("Vision Lab"));
    vl::app::applyTheme(app);

    QCommandLineParser cli;
    cli.setApplicationDescription(QStringLiteral("Interactive 3D geometry and motion from images."));
    cli.addHelpOption();
    QCommandLineOption list(QStringLiteral("list"), QStringLiteral("Print the topic ids and quit."));
    QCommandLineOption module(QStringLiteral("module"), QStringLiteral("Topic to open."), QStringLiteral("id"));
    QCommandLineOption sample(QStringLiteral("sample"), QStringLiteral("Sample index."), QStringLiteral("n"));
    QCommandLineOption set(QStringLiteral("set"), QStringLiteral("Parameter value, key=value (repeatable)."), QStringLiteral("kv"));
    QCommandLineOption view(QStringLiteral("view"), QStringLiteral("View index (1 = first)."), QStringLiteral("n"));
    QCommandLineOption frame(QStringLiteral("frame"), QStringLiteral("Frame index (1 = first)."), QStringLiteral("n"));
    QCommandLineOption click(QStringLiteral("click"), QStringLiteral("Click at image pixel x,y."), QStringLiteral("x,y"));
    QCommandLineOption roi(QStringLiteral("roi"), QStringLiteral("Box x,y,w,h for tracking topics."), QStringLiteral("box"));
    QCommandLineOption shot(QStringLiteral("screenshot"), QStringLiteral("Save the window to a PNG and quit."), QStringLiteral("png"));
    QCommandLineOption size(QStringLiteral("size"), QStringLiteral("Window size WxH."), QStringLiteral("WxH"));
    cli.addOptions({list, module, sample, set, view, frame, click, roi, shot, size});
    cli.process(app);

    // Automated runs start from the default parameters and leave the user's settings alone.
    QTemporaryDir automationSettings;
    if (cli.isSet(shot) || cli.isSet(list)) {
        QSettings::setDefaultFormat(QSettings::IniFormat);
        QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, automationSettings.path());
    }

    vl::app::AppController controller;
    if (cli.isSet(list)) {
        auto& doc = controller.document();
        for (int i = 0; i < doc.count(); ++i)
            std::printf("%-22s %s / %s\n", qPrintable(doc.state(i).info.id), qPrintable(doc.state(i).info.group),
                        qPrintable(doc.state(i).info.title));
        return 0;
    }
    if (cli.isSet(size)) {
        const auto wh = cli.value(size).split(u'x');
        if (wh.size() == 2) controller.window()->resize(wh[0].toInt(), wh[1].toInt());
    }
    controller.start(cli.value(module));

    // Automation: apply the requested state step by step, each after the previous result is ready.
    std::vector<std::function<void()>> steps;
    if (cli.isSet(sample)) steps.push_back([&] { controller.selectSample(cli.value(sample).toInt()); });
    const QStringList sets = cli.values(set);
    if (!sets.isEmpty() || cli.isSet(roi))
        steps.push_back([&, sets] {
            for (const QString& kv : sets) {
                const auto parts = kv.split(u'=');
                if (parts.size() == 2) controller.document().setParam(parts[0], parts[1].toDouble());
            }
            if (cli.isSet(roi)) {
                const auto b = cli.value(roi).split(u',');
                if (b.size() == 4)
                    controller.document().setParam(QStringLiteral("roi"),
                                                   QRectF(b[0].toDouble(), b[1].toDouble(), b[2].toDouble(), b[3].toDouble()));
            }
            controller.syncInspector();
            controller.recompute(0);
        });
    if (cli.isSet(shot)) {
        const QString path = cli.value(shot);
        steps.push_back([&, path] {
            if (cli.isSet(view)) controller.setView(cli.value(view).toInt() - 1);
            if (cli.isSet(frame)) controller.setFrame(cli.value(frame).toInt() - 1);
            if (cli.isSet(click)) {
                const auto xy = cli.value(click).split(u',');
                if (xy.size() == 2) controller.click(QPointF(xy[0].toDouble(), xy[1].toDouble()));
            }
            QTimer::singleShot(150, [&, path] {
                controller.window()->grab().save(path);
                std::printf("saved %s\n", qPrintable(path));
                QApplication::quit();
            });
        });
    }
    if (!steps.empty()) {
        auto runNext = std::make_shared<std::function<void()>>();
        auto index = std::make_shared<size_t>(0);
        *runNext = [&, runNext, index, steps] {
            if (*index >= steps.size()) return;
            const size_t i = (*index)++;
            const bool last = i + 1 == steps.size();
            if (!last) controller.whenIdle(*runNext);
            steps[i]();
            if (last) return;
        };
        controller.whenIdle(*runNext);
        QTimer::singleShot(180000, [] { std::fprintf(stderr, "timeout\n"); QApplication::exit(2); });
    }
    return app.exec();
}
