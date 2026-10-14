#pragma once
// The right panel, read top to bottom: what the topic is, what to try, which input, the
// parameters (built from the module's ParamSpecs), then the numbers the last run produced.

#include <array>
#include <map>
#include <vector>

#include <QScrollArea>

#include "app/model/document.hpp"

class QComboBox;
class QLabel;
class QPushButton;
class QToolButton;
class QVBoxLayout;
class QGridLayout;

namespace vl::app {

class ParamRow;
class Sparkline;

class Inspector : public QScrollArea {
    Q_OBJECT
public:
    explicit Inspector(QWidget* parent = nullptr);

    void setModule(const ModuleState& s);     // rebuild for another topic
    void syncParams(const Params& p);         // reflect values without emitting
    void setResult(const ModuleState& s);     // stats, sparklines, note, error
    void setBoxToolChecked(bool on);
    void setSampleIndex(int index);

signals:
    void paramChanged(const QString& key, const QVariant& value);
    void resetRequested();
    void sampleChosen(int index);
    void openRequested();
    void boxToolToggled(bool on);

private:
    QWidget* section(const QString& title, QWidget* trailing = nullptr);

    QVBoxLayout* layout_;
    QLabel* idea_;
    QLabel* try_;
    QComboBox* samples_;
    QToolButton* open_;
    QPushButton* boxTool_;
    QLabel* boxHint_;
    QWidget* paramsBox_;
    QVBoxLayout* paramsLayout_;
    std::map<QString, ParamRow*> rows_;
    QWidget* statsBox_;
    QGridLayout* statsGrid_;
    QWidget* compareBox_;
    QGridLayout* compareGrid_;
    std::vector<std::array<QLabel*, 3>> compareCells_;
    std::vector<std::pair<QLabel*, QLabel*>> statRows_;
    std::vector<Sparkline*> sparks_;
    QVBoxLayout* seriesLayout_;
    QLabel* note_;
    QLabel* error_;
};

}  // namespace vl::app
