#pragma once
// Left panel: the topics, grouped like the course (Geometry, Stereo, Motion, Flow).

#include <QListWidget>

#include "app/model/document.hpp"

namespace vl::app {

class NavList : public QListWidget {
    Q_OBJECT
public:
    explicit NavList(QWidget* parent = nullptr);
    void populate(const Document& doc);
    void select(int moduleIndex);

signals:
    void moduleChosen(int moduleIndex);
};

}  // namespace vl::app
