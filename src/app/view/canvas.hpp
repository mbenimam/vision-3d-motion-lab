#pragma once
// The picture: zoom (wheel, anchored under the cursor), pan (drag), fit (double-click or F),
// vector overlays that stay one screen pixel thin at any zoom, a click / box tool for modules,
// and a readout of the module's probe under the cursor.

#include <QGraphicsView>

#include "app/module.hpp"

class QGraphicsPixmapItem;
class QGraphicsItemGroup;
class QGraphicsRectItem;
class QLabel;

namespace vl::app {

class Canvas : public QGraphicsView {
    Q_OBJECT
public:
    explicit Canvas(QWidget* parent = nullptr);

    // Show a view. keepZoom: same picture size as before, so the user's zoom and pan are kept.
    void show(const View* view, const std::vector<vl::app::Shape>* extra, bool keepZoom);
    void setBoxTool(bool on);  // drag draws a rectangle instead of panning
    bool boxTool() const { return boxTool_; }
    void setMessage(const QString& text);  // centred message (empty state, error); empty hides it
    void fit();
    void zoomTo(double factor);  // 1.0 = one image pixel per screen pixel
    double zoom() const;
    QImage renderCurrent() const;  // the picture with its overlays, at image resolution

signals:
    void clicked(QPointF pixel);
    void boxDrawn(QRectF pixel);
    void hover(const QString& text);
    void zoomChanged(double factor);

protected:
    void wheelEvent(QWheelEvent* e) override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void mouseDoubleClickEvent(QMouseEvent* e) override;
    void leaveEvent(QEvent* e) override;
    void resizeEvent(QResizeEvent* e) override;
    void drawBackground(QPainter* p, const QRectF& rect) override;
    void drawForeground(QPainter* p, const QRectF& rect) override;

private:
    void rebuildOverlays(const std::vector<vl::app::Shape>& shapes, const std::vector<vl::app::Shape>* extra);
    void updateMessageGeometry();

    QGraphicsScene* scene_;
    QGraphicsPixmapItem* pixmap_;
    QGraphicsItemGroup* overlays_ = nullptr;
    QGraphicsRectItem* box_;
    QLabel* message_;
    std::function<QString(QPoint)> probe_;
    std::optional<Legend> legend_;
    QSize imageSize_;
    bool fitted_ = true;   // stay fitted on resize until the user zooms
    bool boxTool_ = false;
    bool dragging_ = false;
    QPoint pressPos_;
    QPointF boxStart_;
};

}  // namespace vl::app
