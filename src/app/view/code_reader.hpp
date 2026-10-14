#pragma once
// "Read the code": the from-scratch implementation of the current topic, next to its picture.
// The core sources are embedded in the executable (CMake generates sources.qrc), so the reader
// always shows exactly the code that produced the result on screen.

#include <QDialog>

class QListWidget;
class QPlainTextEdit;
class QLabel;

namespace vl::app {

class CodeReader : public QDialog {
    Q_OBJECT
public:
    explicit CodeReader(QWidget* parent = nullptr);
    // Show the topic's own files first, then every other file of the core library.
    void showTopic(const QString& topicTitle, const QStringList& sources);

private:
    void openFile(const QString& path);

    QListWidget* files_;
    QPlainTextEdit* text_;
    QLabel* path_;
};

}  // namespace vl::app
