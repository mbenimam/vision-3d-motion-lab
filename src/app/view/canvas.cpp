#include "app/view/canvas.hpp"

#include <cmath>

#include <QGraphicsEllipseItem>
#include <QGraphicsItemGroup>
#include <QGraphicsLineItem>
#include <QGraphicsPathItem>
#include <QGraphicsPixmapItem>
#include <QGraphicsPolygonItem>
#include <QGraphicsRectItem>
#include <QGraphicsScene>
#include <QGraphicsSimpleTextItem>
#include <QLabel>
#include <QMouseEvent>
#include <QPainterPath>
#include <QScrollBar>
#include <QWheelEvent>

#include "core/common/image.hpp"

namespace vl::app {
namespace {

QPen cosmetic(const Pen& p) {
    QPen pen(p.color, p.width);
    pen.setCosmetic(true);  // width in screen pixels, whatever the zoom
    pen.setCapStyle(Qt::RoundCap);
    pen.setJoinStyle(Qt::RoundJoin);
    if (p.dashed) pen.setStyle(Qt::DashLine);
    return pen;
}

// A marker that keeps its screen size: drawn in item coordinates with ItemIgnoresTransformations.
class Dot : public QGraphicsEllipseItem {
public:
    Dot(QPointF at, double r, QColor c) : QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r) {
        setPos(at);
        setFlag(ItemIgnoresTransformations);
        setBrush(c);
        QPen outline(QColor(0, 0, 0, 160), 1);
        outline.setCosmetic(true);
        setPen(outline);
    }
};

}  // namespace

Canvas::Canvas(QWidget* parent) : QGraphicsView(parent), scene_(new QGraphicsScene(this)) {
    setObjectName(QStringLiteral("canvas"));
    setScene(scene_);
    setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);
    setTransformationAnchor(AnchorUnderMouse);
    setResizeAnchor(AnchorViewCenter);
    setViewportUpdateMode(SmartViewportUpdate);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setFrameShape(NoFrame);
    setMouseTracking(true);
    setDragMode(NoDrag);

    pixmap_ = scene_->addPixmap({});
    pixmap_->setTransformationMode(Qt::SmoothTransformation);
    box_ = scene_->addRect({}, cosmetic({colors::accent, 1.5, true}), QColor(255, 176, 0, 30));
    box_->setZValue(10);
    box_->hide();

    message_ = new QLabel(viewport());
    message_->setObjectName(QStringLiteral("canvasMessage"));
    message_->setAlignment(Qt::AlignCenter);
    message_->setWordWrap(true);
    message_->hide();
}

void Canvas::show(const View* view, const std::vector<vl::app::Shape>* extra, bool keepZoom) {
    if (!view) {
        pixmap_->setPixmap({});
        rebuildOverlays({}, nullptr);
        probe_ = {};
        legend_.reset();
        handles_.clear();
        rebuildHandles();
        return;
    }
    const QImage picture = view->picture();
    const QSize newSize = picture.size();
    pixmap_->setPixmap(QPixmap::fromImage(picture));
    // Sharp pixels when zoomed in: that is how one sees a disparity step or a block of the flow.
    pixmap_->setTransformationMode(zoom() > 2.0 ? Qt::FastTransformation : Qt::SmoothTransformation);
    scene_->setSceneRect(QRectF(QPointF(0, 0), QSizeF(newSize)).adjusted(-40, -40, 40, 40));
    rebuildOverlays(view->shapes, extra);
    probe_ = view->probe;
    legend_ = view->legend;
    if (draggedHandle_ < 0) {  // never replace the points under the user's mouse
        handles_ = view->handles;
        rebuildHandles();
    }
    viewport()->update();
    if (!keepZoom || newSize != imageSize_ || fitted_) {
        imageSize_ = newSize;
        fit();
    }
}

void Canvas::rebuildOverlays(const std::vector<vl::app::Shape>& shapes, const std::vector<vl::app::Shape>* extra) {
    delete overlays_;  // deletes every shape item with it
    overlays_ = new QGraphicsItemGroup;
    overlays_->setZValue(5);
    scene_->addItem(overlays_);

    auto add = [&](const vl::app::Shape& shape) {
        std::visit(
            [&](const auto& s) {
                using T = std::decay_t<decltype(s)>;
                if constexpr (std::is_same_v<T, LineShape>) {
                    auto* it = new QGraphicsLineItem(s.line);
                    it->setPen(cosmetic(s.pen));
                    overlays_->addToGroup(it);
                } else if constexpr (std::is_same_v<T, PolylineShape>) {
                    QPainterPath path;
                    if (!s.points.isEmpty()) {
                        path.addPolygon(s.points);
                        if (s.closed) path.closeSubpath();
                    }
                    auto* it = new QGraphicsPathItem(path);
                    it->setPen(cosmetic(s.pen));
                    overlays_->addToGroup(it);
                } else if constexpr (std::is_same_v<T, PointsShape>) {
                    for (const QPointF& p : s.points) overlays_->addToGroup(new Dot(p, s.radius, s.color));
                } else if constexpr (std::is_same_v<T, RectShape>) {
                    auto* it = new QGraphicsRectItem(s.rect);
                    it->setPen(cosmetic(s.pen));
                    overlays_->addToGroup(it);
                } else if constexpr (std::is_same_v<T, EllipseShape>) {
                    auto* it = new QGraphicsEllipseItem(-s.rx, -s.ry, 2 * s.rx, 2 * s.ry);
                    it->setPos(s.centre);
                    it->setRotation(s.angleDeg);
                    it->setPen(cosmetic(s.pen));
                    overlays_->addToGroup(it);
                } else if constexpr (std::is_same_v<T, ArrowsShape>) {
                    QPainterPath path;
                    for (const QLineF& l : s.arrows) {
                        path.moveTo(l.p1());
                        path.lineTo(l.p2());
                        const double len = l.length();
                        if (len < 1e-6) continue;
                        const double head = std::min(4.0, 0.35 * len);
                        const QPointF dir = (l.p2() - l.p1()) / len, nrm(-dir.y(), dir.x());
                        path.moveTo(l.p2() - head * dir + 0.6 * head * nrm);
                        path.lineTo(l.p2());
                        path.lineTo(l.p2() - head * dir - 0.6 * head * nrm);
                    }
                    auto* it = new QGraphicsPathItem(path);
                    it->setPen(cosmetic(s.pen));
                    overlays_->addToGroup(it);
                } else if constexpr (std::is_same_v<T, LabelShape>) {
                    auto* it = new QGraphicsSimpleTextItem(s.text);
                    it->setBrush(s.color);
                    QPen halo(QColor(0, 0, 0, 200), 3);
                    halo.setCosmetic(true);
                    it->setPen(Qt::NoPen);
                    it->setFlag(QGraphicsItem::ItemIgnoresTransformations);
                    it->setPos(s.at);
                    QFont f = font();
                    f.setPointSizeF(f.pointSizeF() * 0.95);
                    f.setBold(true);
                    it->setFont(f);
                    // a dark plate behind the text keeps it readable on any picture
                    auto* plate = new QGraphicsRectItem(it->boundingRect().adjusted(-3, -1, 3, 1));
                    plate->setBrush(QColor(14, 17, 23, 200));
                    plate->setPen(Qt::NoPen);
                    plate->setFlag(QGraphicsItem::ItemIgnoresTransformations);
                    plate->setPos(s.at);
                    overlays_->addToGroup(plate);
                    overlays_->addToGroup(it);
                }
            },
            shape);
    };
    for (const auto& s : shapes) add(s);
    if (extra)
        for (const auto& s : *extra) add(s);
}

// Handles: a closed outline through the points and a ring on each, drawn at a fixed screen size.
void Canvas::rebuildHandles() {
    delete handleItems_;
    handleItems_ = nullptr;
    if (handles_.empty()) return;
    handleItems_ = new QGraphicsItemGroup;
    handleItems_->setZValue(20);
    scene_->addItem(handleItems_);
    QPolygonF outline;
    for (const QPointF& h : handles_) outline << h;
    auto* poly = new QGraphicsPolygonItem(outline);
    poly->setPen(cosmetic({colors::accent, 2.0, false}));
    poly->setBrush(QColor(255, 176, 0, 28));
    handleItems_->addToGroup(poly);
    for (const QPointF& h : handles_) {
        auto* ring = new QGraphicsEllipseItem(-8, -8, 16, 16);
        ring->setPos(h);
        ring->setFlag(QGraphicsItem::ItemIgnoresTransformations);
        ring->setBrush(QColor(14, 17, 23, 180));
        QPen pen(colors::accent, 2.5);
        pen.setCosmetic(true);
        ring->setPen(pen);
        handleItems_->addToGroup(ring);
    }
}

int Canvas::handleAt(QPoint viewportPos) const {
    for (int i = 0; i < int(handles_.size()); ++i)
        if (QLineF(QPointF(mapFromScene(handles_[size_t(i)])), QPointF(viewportPos)).length() <= 12) return i;
    return -1;
}

void Canvas::setBoxTool(bool on) {
    boxTool_ = on;
    viewport()->setCursor(on ? Qt::CrossCursor : Qt::ArrowCursor);
}

void Canvas::setMessage(const QString& text) {
    message_->setText(text);
    message_->setVisible(!text.isEmpty());
    updateMessageGeometry();
}

void Canvas::updateMessageGeometry() {
    const int w = std::min(460, viewport()->width() - 40);
    message_->setFixedWidth(std::max(120, w));
    message_->adjustSize();
    message_->move((viewport()->width() - message_->width()) / 2, (viewport()->height() - message_->height()) / 2);
}

void Canvas::fit() {
    if (imageSize_.isEmpty()) return;
    const QRectF r(QPointF(0, 0), QSizeF(imageSize_));
    fitInView(r.adjusted(-8, -8, 8, 8), Qt::KeepAspectRatio);
    fitted_ = true;
    pixmap_->setTransformationMode(zoom() > 2.0 ? Qt::FastTransformation : Qt::SmoothTransformation);
    emit zoomChanged(zoom());
}

void Canvas::zoomTo(double factor) {
    const double z = zoom();
    if (z <= 0) return;
    scale(factor / z, factor / z);
    fitted_ = false;
    pixmap_->setTransformationMode(zoom() > 2.0 ? Qt::FastTransformation : Qt::SmoothTransformation);
    emit zoomChanged(zoom());
}

double Canvas::zoom() const { return transform().m11(); }

QImage Canvas::renderCurrent() const {
    if (imageSize_.isEmpty()) return {};
    QImage out(imageSize_, QImage::Format_ARGB32_Premultiplied);
    out.fill(Qt::black);
    QPainter p(&out);
    p.setRenderHint(QPainter::Antialiasing);
    const QRectF r(QPointF(0, 0), QSizeF(imageSize_));
    scene_->render(&p, QRectF(out.rect()), r);
    return out;
}

void Canvas::wheelEvent(QWheelEvent* e) {
    const double steps = e->angleDelta().y() / 120.0;
    const double factor = std::pow(1.2, steps);
    const double z = zoom() * factor;
    if (z < 0.05 || z > 64) return;
    scale(factor, factor);
    fitted_ = false;
    pixmap_->setTransformationMode(zoom() > 2.0 ? Qt::FastTransformation : Qt::SmoothTransformation);
    emit zoomChanged(zoom());
    e->accept();
}

void Canvas::mousePressEvent(QMouseEvent* e) {
    if (e->button() != Qt::LeftButton) return QGraphicsView::mousePressEvent(e);
    pressPos_ = e->position().toPoint();
    dragging_ = false;
    draggedHandle_ = boxTool_ ? -1 : handleAt(pressPos_);
    if (boxTool_) {
        boxStart_ = mapToScene(pressPos_);
        box_->setRect(QRectF(boxStart_, boxStart_));
        box_->show();
    }
    e->accept();
}

void Canvas::mouseMoveEvent(QMouseEvent* e) {
    const QPoint pos = e->position().toPoint();
    if (e->buttons() & Qt::LeftButton) {
        if (!dragging_ && (pos - pressPos_).manhattanLength() > 4) dragging_ = true;
        if (draggedHandle_ >= 0) {
            const QPointF p = mapToScene(pos);
            handles_[size_t(draggedHandle_)] = QPointF(std::clamp(p.x(), 0.0, double(imageSize_.width())),
                                                       std::clamp(p.y(), 0.0, double(imageSize_.height())));
            rebuildHandles();
        } else if (dragging_ && boxTool_) {
            box_->setRect(QRectF(boxStart_, mapToScene(pos)).normalized());
        } else if (dragging_) {
            const QPoint delta = pos - pressPos_;
            pressPos_ = pos;
            horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.x());
            verticalScrollBar()->setValue(verticalScrollBar()->value() - delta.y());
            fitted_ = false;
            viewport()->setCursor(Qt::ClosedHandCursor);
        }
    }
    if (!(e->buttons() & Qt::LeftButton) && !boxTool_)
        viewport()->setCursor(handleAt(pos) >= 0 ? Qt::OpenHandCursor : Qt::ArrowCursor);
    const QPointF p = mapToScene(pos);
    const QPoint px(int(std::floor(p.x())), int(std::floor(p.y())));
    if (!imageSize_.isEmpty() && QRect(QPoint(0, 0), imageSize_).contains(px)) {
        QString text = QStringLiteral("x %1  y %2").arg(px.x()).arg(px.y());
        if (probe_) {
            const QString extra = probe_(px);
            if (!extra.isEmpty()) text += QStringLiteral("  ·  ") + extra;
        }
        emit hover(text);
    } else {
        emit hover({});
    }
}

void Canvas::mouseReleaseEvent(QMouseEvent* e) {
    if (e->button() != Qt::LeftButton) return QGraphicsView::mouseReleaseEvent(e);
    viewport()->setCursor(boxTool_ ? Qt::CrossCursor : Qt::ArrowCursor);
    const QPointF p = mapToScene(e->position().toPoint());
    if (draggedHandle_ >= 0) {
        draggedHandle_ = -1;
        if (dragging_) emit handlesMoved(handles_);
        dragging_ = false;
        return;
    }
    if (boxTool_) {
        box_->hide();
        const QRectF r = QRectF(boxStart_, p).normalized() & QRectF(QPointF(0, 0), QSizeF(imageSize_));
        if (dragging_ && r.width() >= 4 && r.height() >= 4) emit boxDrawn(r);
    } else if (!dragging_) {
        if (QRectF(QPointF(0, 0), QSizeF(imageSize_)).contains(p)) emit clicked(p);
    }
    dragging_ = false;
}

void Canvas::mouseDoubleClickEvent(QMouseEvent*) { fit(); }

void Canvas::leaveEvent(QEvent* e) {
    emit hover({});
    QGraphicsView::leaveEvent(e);
}

void Canvas::resizeEvent(QResizeEvent* e) {
    QGraphicsView::resizeEvent(e);
    if (fitted_) fit();
    updateMessageGeometry();
}

void Canvas::drawBackground(QPainter* p, const QRectF& rect) {
    p->fillRect(rect, QColor(0x0b, 0x0e, 0x14));
    // a faint border around the picture, so dark images still show their edges
    if (!imageSize_.isEmpty()) {
        QPen pen(QColor(0x22, 0x2a, 0x37), 1);
        pen.setCosmetic(true);
        p->setPen(pen);
        p->drawRect(QRectF(QPointF(0, 0), QSizeF(imageSize_)).adjusted(-0.5, -0.5, 0.5, 0.5));
    }
}

// The legend is drawn in screen space (bottom-left corner), so it never zooms with the picture.
void Canvas::drawForeground(QPainter* p, const QRectF&) {
    if (!legend_) return;
    p->save();
    p->resetTransform();
    p->setRenderHint(QPainter::Antialiasing);
    QFont f = font();
    f.setPointSizeF(f.pointSizeF() * 0.85);
    p->setFont(f);
    if (!legend_->key.isNull()) {  // a picture key: drawn square, label underneath
        const int side = 96, pad = 10;
        const int w = std::max(side, QFontMetrics(f).horizontalAdvance(legend_->label));
        const QRect box(14, viewport()->height() - side - 44, w + 2 * pad, side + 34);
        p->setPen(QColor(0x2c, 0x35, 0x46));
        p->setBrush(QColor(14, 17, 23, 215));
        p->drawRoundedRect(box, 6, 6);
        p->drawImage(QRect(box.left() + pad + (w - side) / 2, box.top() + pad, side, side), legend_->key);
        p->setPen(QColor(0x9a, 0xa4, 0xb2));
        p->drawText(QRect(box.left() + pad, box.bottom() - 22, w, 16), Qt::AlignCenter, legend_->label);
        p->restore();
        return;
    }
    const int w = std::max(180, QFontMetrics(f).horizontalAdvance(legend_->label) + 4), barH = 8, pad = 10;
    const QRect box(14, viewport()->height() - 64, w + 2 * pad, 50);
    p->setPen(QColor(0x2c, 0x35, 0x46));
    p->setBrush(QColor(14, 17, 23, 215));
    p->drawRoundedRect(box, 6, 6);
    static const QImage bar = [] {
        cv::Mat ramp(1, 256, CV_32F);
        for (int i = 0; i < 256; ++i) ramp.at<float>(0, i) = float(i);
        return toQImage(vl::colorize(ramp, 0, 255));
    }();
    const QRect barRect(box.left() + pad, box.top() + 24, w, barH);
    p->drawImage(barRect, bar);
    p->setPen(QColor(0x9a, 0xa4, 0xb2));
    p->drawText(QRect(box.left() + pad, box.top() + 5, w, 16), Qt::AlignLeft | Qt::AlignVCenter, legend_->label);
    p->setPen(QColor(0xd6, 0xdc, 0xe4));
    const QString lo = legend_->loText.isEmpty() ? QString::number(legend_->lo, 'g', 3) : legend_->loText;
    const QString hi = legend_->hiText.isEmpty() ? QString::number(legend_->hi, 'g', 3) : legend_->hiText;
    p->drawText(QRect(barRect.left(), barRect.bottom() + 2, w / 2, 14), Qt::AlignLeft | Qt::AlignTop, lo);
    p->drawText(QRect(barRect.left() + w / 2, barRect.bottom() + 2, w / 2, 14), Qt::AlignRight | Qt::AlignTop, hi);
    p->restore();
}

}  // namespace vl::app
