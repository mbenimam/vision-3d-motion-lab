#include "app/view/code_reader.hpp"

#include <QDirIterator>
#include <QFileInfo>
#include <QFile>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QPainter>
#include <QPlainTextEdit>
#include <QRegularExpression>
#include <QSyntaxHighlighter>
#include <QTextBlock>
#include <QVBoxLayout>

namespace vl::app {
namespace {

// Just enough C++ colouring to read comfortably: comments (where the maths is explained) stand out.
class CppHighlighter : public QSyntaxHighlighter {
public:
    explicit CppHighlighter(QTextDocument* doc) : QSyntaxHighlighter(doc) {
        auto fmt = [](const char* colour, bool bold = false, bool italic = false) {
            QTextCharFormat f;
            f.setForeground(QColor(colour));
            if (bold) f.setFontWeight(QFont::DemiBold);
            f.setFontItalic(italic);
            return f;
        };
        const QStringList keywords = {"auto", "bool", "break", "case", "class", "const", "constexpr", "continue",
                                      "default", "double", "else", "enum", "false", "float", "for", "if", "inline",
                                      "int", "long", "namespace", "nullptr", "operator", "private", "public",
                                      "return", "static", "struct", "switch", "template", "this", "throw", "true",
                                      "typename", "using", "void", "while", "size_t", "std", "cv"};
        rules_.push_back({QRegularExpression(QStringLiteral("\\b(%1)\\b").arg(keywords.join(u'|'))), fmt("#c792ea")});
        rules_.push_back({QRegularExpression(QStringLiteral("\\b[0-9]+(\\.[0-9]+)?f?\\b")), fmt("#f78c6c")});
        rules_.push_back({QRegularExpression(QStringLiteral("\\b[A-Z][A-Za-z0-9_]*\\b")), fmt("#82aaff")});
        rules_.push_back({QRegularExpression(QStringLiteral("^\\s*#\\s*\\w+.*$")), fmt("#89ddff")});
        rules_.push_back({QRegularExpression(QStringLiteral("\"[^\"\\n]*\"")), fmt("#c3e88d")});
        rules_.push_back({QRegularExpression(QStringLiteral("//[^\\n]*")), fmt("#e3c98f", false, true)});
    }

protected:
    void highlightBlock(const QString& text) override {
        for (const auto& r : rules_) {
            auto it = r.pattern.globalMatch(text);
            while (it.hasNext()) {
                const auto m = it.next();
                setFormat(int(m.capturedStart()), int(m.capturedLength()), r.format);
            }
        }
    }

private:
    struct Rule {
        QRegularExpression pattern;
        QTextCharFormat format;
    };
    std::vector<Rule> rules_;
};

// A plain-text editor with a line-number gutter (so the README and the course can cite lines).
class CodeView : public QPlainTextEdit {
public:
    explicit CodeView(QWidget* parent = nullptr) : QPlainTextEdit(parent), gutter_(new Gutter(this)) {
        connect(this, &QPlainTextEdit::blockCountChanged, this, [this] { setViewportMargins(gutterWidth(), 0, 0, 0); });
        connect(this, &QPlainTextEdit::updateRequest, this, [this](const QRect& r, int dy) {
            dy ? gutter_->scroll(0, dy) : gutter_->update(0, r.y(), gutter_->width(), r.height());
        });
        setViewportMargins(gutterWidth(), 0, 0, 0);
    }
    int gutterWidth() const { return 16 + fontMetrics().horizontalAdvance(QStringLiteral("0000")); }

protected:
    void resizeEvent(QResizeEvent* e) override {
        QPlainTextEdit::resizeEvent(e);
        const QRect cr = contentsRect();
        gutter_->setGeometry(QRect(cr.left(), cr.top(), gutterWidth(), cr.height()));
    }

private:
    class Gutter : public QWidget {
    public:
        explicit Gutter(CodeView* view) : QWidget(view), view_(view) {}

    protected:
        void paintEvent(QPaintEvent* e) override {
            QPainter p(this);
            p.fillRect(e->rect(), QColor(0x0b, 0x0e, 0x14));
            p.setPen(QColor(0x5a, 0x63, 0x70));
            p.setFont(view_->font());
            QTextBlock block = view_->firstVisibleBlock();
            int top = qRound(view_->blockBoundingGeometry(block).translated(view_->contentOffset()).top());
            while (block.isValid() && top <= e->rect().bottom()) {
                const int h = qRound(view_->blockBoundingRect(block).height());
                p.drawText(0, top, width() - 8, h, Qt::AlignRight, QString::number(block.blockNumber() + 1));
                top += h;
                block = block.next();
            }
        }

    private:
        CodeView* view_;
    };
    Gutter* gutter_;
};

}  // namespace

CodeReader::CodeReader(QWidget* parent) : QDialog(parent) {
    setObjectName(QStringLiteral("codeReader"));
    setWindowTitle(tr("Read the code"));
    resize(1180, 820);
    auto* h = new QHBoxLayout(this);
    h->setContentsMargins(0, 0, 0, 0);
    h->setSpacing(0);

    files_ = new QListWidget;
    files_->setObjectName(QStringLiteral("codeFiles"));
    files_->setFixedWidth(270);
    h->addWidget(files_);

    auto* right = new QWidget;
    auto* v = new QVBoxLayout(right);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(0);
    path_ = new QLabel;
    path_->setObjectName(QStringLiteral("codePath"));
    v->addWidget(path_);
    auto* view = new CodeView;
    text_ = view;
    text_->setObjectName(QStringLiteral("code"));
    text_->setReadOnly(true);
    text_->setLineWrapMode(QPlainTextEdit::NoWrap);
    QFont mono(QStringLiteral("JetBrains Mono"));
    mono.setStyleHint(QFont::Monospace);
    mono.setPointSizeF(10.5);
    text_->setFont(mono);
    new CppHighlighter(text_->document());
    v->addWidget(text_, 1);
    h->addWidget(right, 1);

    connect(files_, &QListWidget::currentItemChanged, this, [this](QListWidgetItem* item) {
        if (item && item->data(Qt::UserRole).isValid()) openFile(item->data(Qt::UserRole).toString());
    });
}

void CodeReader::showTopic(const QString& topicTitle, const QStringList& sources) {
    setWindowTitle(tr("Read the code · %1").arg(topicTitle));
    const QSignalBlocker block(files_);
    files_->clear();
    auto header = [this](const QString& text) {
        auto* item = new QListWidgetItem(text.toUpper());
        item->setFlags(Qt::NoItemFlags);
        item->setForeground(QColor(0x9a, 0xa4, 0xb2));
        files_->addItem(item);
    };
    auto add = [this](const QString& path) {
        auto* item = new QListWidgetItem(path.section(u'/', -2));
        item->setData(Qt::UserRole, path);
        item->setToolTip(path);
        files_->addItem(item);
    };
    header(tr("This topic"));
    for (const QString& s : sources) add(s);
    header(tr("The whole core library"));
    QStringList all;
    QDirIterator it(QStringLiteral(":/source/src/core"), QDirIterator::Subdirectories);
    while (it.hasNext()) {
        const QString p = it.next().mid(QStringLiteral(":/source/").size());
        if (!sources.contains(p) && !p.endsWith(QLatin1String("empty.cpp"))) all << p;
    }
    all.sort();
    for (const QString& p : all) add(p);
    if (files_->count() > 1) files_->setCurrentRow(1);
    openFile(sources.isEmpty() ? all.value(0) : sources.front());
}

void CodeReader::openFile(const QString& path) {
    QFile f(QStringLiteral(":/source/") + path);
    path_->setText(path);
    text_->setPlainText(f.open(QIODevice::ReadOnly) ? QString::fromUtf8(f.readAll())
                                                    : tr("// %1 is not embedded in this build").arg(path));
}

}  // namespace vl::app
