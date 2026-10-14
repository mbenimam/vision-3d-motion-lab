#include "app/view/nav_list.hpp"

namespace vl::app {

namespace {
constexpr int kModuleRole = Qt::UserRole + 1;
}

NavList::NavList(QWidget* parent) : QListWidget(parent) {
    setObjectName(QStringLiteral("nav"));
    setFrameShape(NoFrame);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setFocusPolicy(Qt::StrongFocus);  // arrow keys move between topics
    setUniformItemSizes(false);
    connect(this, &QListWidget::currentItemChanged, this, [this](QListWidgetItem* item) {
        if (item && item->data(kModuleRole).isValid()) emit moduleChosen(item->data(kModuleRole).toInt());
    });
}

void NavList::populate(const Document& doc) {
    const QSignalBlocker block(this);
    clear();
    QString group;
    for (int i = 0; i < doc.count(); ++i) {
        const auto& info = doc.state(i).info;
        if (info.group != group) {
            group = info.group;
            auto* header = new QListWidgetItem(group.toUpper());
            header->setFlags(Qt::NoItemFlags);  // not selectable, skipped by the arrow keys
            header->setData(Qt::UserRole, QStringLiteral("header"));
            QFont f = font();
            f.setPointSizeF(f.pointSizeF() * 0.82);
            f.setBold(true);
            f.setLetterSpacing(QFont::PercentageSpacing, 108);
            header->setFont(f);
            header->setForeground(QColor(0x9a, 0xa4, 0xb2));
            header->setSizeHint(QSize(0, i == 0 ? 26 : 38));
            header->setTextAlignment(Qt::AlignLeft | Qt::AlignBottom);
            addItem(header);
        }
        auto* item = new QListWidgetItem(info.title);
        item->setData(kModuleRole, i);
        item->setToolTip(info.group + QStringLiteral(" · ") + info.title);
        item->setSizeHint(QSize(0, 34));
        addItem(item);
    }
}

void NavList::select(int moduleIndex) {
    for (int r = 0; r < count(); ++r)
        if (item(r)->data(kModuleRole).toInt() == moduleIndex && item(r)->data(kModuleRole).isValid()) {
            const QSignalBlocker block(this);
            setCurrentRow(r);
            return;
        }
}

}  // namespace vl::app
