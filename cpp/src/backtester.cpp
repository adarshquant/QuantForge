#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct Record {
    std::string date;
    std::string symbol;
    double return_1d;
    double predicted_alpha;
    double prediction_rank;
    double position_weight;
};

struct DailyResult {
    std::string date;
    double gross_return;
    double turnover;
    double transaction_cost;
    double slippage_cost;
    double net_return;
    double equity;
    double drawdown;
};

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;

    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }

    return fields;
}

double to_double(const std::string& value) {
    return std::stod(value);
}

int main() {
    const std::string input_path =
        "data/processed/final_backtest_input.csv";

    const std::string output_path =
        "data/processed/cpp_equity_curve.csv";

    const double initial_capital = 1000000.0;
    const double transaction_cost_rate = 0.0005;
    const double slippage_rate = 0.0005;

    std::ifstream file(input_path);

    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open input file: "
                  << input_path << std::endl;
        return 1;
    }

    std::string line;

    if (!std::getline(file, line)) {
        std::cerr << "ERROR: Input file is empty." << std::endl;
        return 1;
    }

    std::vector<Record> records;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        auto fields = split_csv(line);

        if (fields.size() < 6) {
            continue;
        }

        try {
            Record record;

            record.date = fields[0];
            record.symbol = fields[1];
            record.return_1d = to_double(fields[2]);
            record.predicted_alpha = to_double(fields[3]);
            record.prediction_rank = to_double(fields[4]);
            record.position_weight = to_double(fields[5]);

            records.push_back(record);
        }
        catch (...) {
            continue;
        }
    }

    file.close();

    if (records.empty()) {
        std::cerr << "ERROR: No valid records found." << std::endl;
        return 1;
    }

    std::sort(
        records.begin(),
        records.end(),
        [](const Record& a, const Record& b) {
            if (a.date != b.date) {
                return a.date < b.date;
            }

            return a.symbol < b.symbol;
        }
    );

    std::map<std::string, std::vector<Record>> daily_records;

    for (const auto& record : records) {
        daily_records[record.date].push_back(record);
    }

    std::unordered_map<std::string, double> previous_weights;

    std::vector<DailyResult> results;

    double equity = initial_capital;
    double peak_equity = initial_capital;

    double total_turnover = 0.0;
    double total_transaction_cost = 0.0;
    double total_slippage_cost = 0.0;

    int winning_days = 0;
    int losing_days = 0;

    for (const auto& [date, day_records] : daily_records) {
        std::unordered_map<std::string, double> current_weights;

        double gross_return = 0.0;

        for (const auto& record : day_records) {
            current_weights[record.symbol] = record.position_weight;

            gross_return +=
                record.position_weight * record.return_1d;
        }

        std::unordered_map<std::string, bool> symbols;

        for (const auto& [symbol, weight] : previous_weights) {
            symbols[symbol] = true;
        }

        for (const auto& [symbol, weight] : current_weights) {
            symbols[symbol] = true;
        }

        double turnover = 0.0;

        for (const auto& [symbol, exists] : symbols) {
            double previous_weight = 0.0;
            double current_weight = 0.0;

            auto previous_it = previous_weights.find(symbol);

            if (previous_it != previous_weights.end()) {
                previous_weight = previous_it->second;
            }

            auto current_it = current_weights.find(symbol);

            if (current_it != current_weights.end()) {
                current_weight = current_it->second;
            }

            turnover +=
                std::abs(current_weight - previous_weight);
        }

        double transaction_cost =
            turnover * transaction_cost_rate;

        double slippage_cost =
            turnover * slippage_rate;

        double net_return =
            gross_return -
            transaction_cost -
            slippage_cost;

        equity *= (1.0 + net_return);

        peak_equity = std::max(
            peak_equity,
            equity
        );

        double drawdown =
            equity / peak_equity - 1.0;

        if (net_return > 0.0) {
            winning_days++;
        }
        else if (net_return < 0.0) {
            losing_days++;
        }

        total_turnover += turnover;
        total_transaction_cost +=
            equity * transaction_cost;
        total_slippage_cost +=
            equity * slippage_cost;

        DailyResult result;

        result.date = date;
        result.gross_return = gross_return;
        result.turnover = turnover;
        result.transaction_cost = transaction_cost;
        result.slippage_cost = slippage_cost;
        result.net_return = net_return;
        result.equity = equity;
        result.drawdown = drawdown;

        results.push_back(result);

        previous_weights = current_weights;
    }

    std::vector<double> returns;

    for (const auto& result : results) {
        returns.push_back(result.net_return);
    }

    double total_return =
        equity / initial_capital - 1.0;

    double mean_return = 0.0;

    for (double value : returns) {
        mean_return += value;
    }

    mean_return /= returns.size();

    double variance = 0.0;

    for (double value : returns) {
        variance +=
            std::pow(value - mean_return, 2);
    }

    variance /= returns.size();

    double daily_volatility =
        std::sqrt(variance);

    double annualized_volatility =
        daily_volatility * std::sqrt(252.0);

    double sharpe = 0.0;

    if (daily_volatility > 0.0) {
        sharpe =
            mean_return /
            daily_volatility *
            std::sqrt(252.0);
    }

    double max_drawdown = 0.0;

    for (const auto& result : results) {
        max_drawdown =
            std::min(
                max_drawdown,
                result.drawdown
            );
    }

    double win_rate = 0.0;

    if (!returns.empty()) {
        win_rate =
            static_cast<double>(winning_days) /
            static_cast<double>(returns.size());
    }

    double gross_profit = 0.0;
    double gross_loss = 0.0;

    for (double value : returns) {
        if (value > 0.0) {
            gross_profit += value;
        }
        else if (value < 0.0) {
            gross_loss += std::abs(value);
        }
    }

    double profit_factor = 0.0;

    if (gross_loss > 0.0) {
        profit_factor =
            gross_profit / gross_loss;
    }

    std::ofstream output(output_path);

    if (!output.is_open()) {
        std::cerr << "ERROR: Could not create output file."
                  << std::endl;
        return 1;
    }

    output
        << "Date,Gross_Return,Turnover,Transaction_Cost,"
        << "Slippage_Cost,Net_Return,Equity,Drawdown\n";

    output << std::fixed << std::setprecision(10);

    for (const auto& result : results) {
        output
            << result.date << ","
            << result.gross_return << ","
            << result.turnover << ","
            << result.transaction_cost << ","
            << result.slippage_cost << ","
            << result.net_return << ","
            << result.equity << ","
            << result.drawdown << "\n";
    }

    output.close();

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "============================================================"
              << std::endl;
    std::cout << "QUANTFORGE C++ RISK-AWARE BACKTESTING ENGINE"
              << std::endl;
    std::cout << "============================================================"
              << std::endl;

    std::cout << "Trading sessions       : "
              << results.size()
              << std::endl;

    std::cout << "Position rows          : "
              << records.size()
              << std::endl;

    std::cout << "Initial capital        : ₹"
              << initial_capital
              << std::endl;

    std::cout << "Final equity           : ₹"
              << equity
              << std::endl;

    std::cout << "Total return           : "
              << total_return * 100.0
              << "%"
              << std::endl;

    std::cout << "Annualized volatility  : "
              << annualized_volatility * 100.0
              << "%"
              << std::endl;

    std::cout << "Max drawdown           : "
              << max_drawdown * 100.0
              << "%"
              << std::endl;

    std::cout << "Sharpe ratio           : "
              << sharpe
              << std::endl;

    std::cout << "Win rate               : "
              << win_rate * 100.0
              << "%"
              << std::endl;

    std::cout << "Profit factor          : "
              << profit_factor
              << std::endl;

    std::cout << "Total turnover         : "
              << total_turnover * 100.0
              << "%"
              << std::endl;

    std::cout << "Transaction cost rate  : "
              << transaction_cost_rate * 100.0
              << "%"
              << std::endl;

    std::cout << "Slippage rate          : "
              << slippage_rate * 100.0
              << "%"
              << std::endl;

    std::cout << "Equity curve output    : "
              << output_path
              << std::endl;

    std::cout << std::endl;

    std::cout
        << "Risk-aware backtest completed successfully."
        << std::endl;

    return 0;
}