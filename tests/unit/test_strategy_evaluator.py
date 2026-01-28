"""
Unit tests for Strategy Evaluator
"""
import pytest
from datetime import datetime, date, timedelta


class TestStrategyGrade:
    """Test StrategyGrade enum"""

    def test_grade_values(self):
        from src.model.strategy_evaluator import StrategyGrade

        assert StrategyGrade.A_PLUS.value == "A+"
        assert StrategyGrade.F.value == "F"


class TestTradeRecord:
    """Test TradeRecord class"""

    def test_trade_creation(self):
        from src.model.strategy_evaluator import TradeRecord

        trade = TradeRecord(
            trade_id="test-1",
            symbol="TQQQ",
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=50.0,
            exit_price=51.0,
            quantity=100,
            side="long",
            pnl=100.0
        )

        assert trade.is_winner is True
        assert trade.is_closed is True

    def test_loser_trade(self):
        from src.model.strategy_evaluator import TradeRecord

        trade = TradeRecord(
            trade_id="test-2",
            symbol="TQQQ",
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=50.0,
            exit_price=49.0,
            quantity=100,
            side="long",
            pnl=-100.0
        )

        assert trade.is_winner is False


class TestStrategyEvaluator:
    """Test StrategyEvaluator class"""

    @pytest.fixture
    def evaluator(self):
        from src.model.strategy_evaluator import StrategyEvaluator
        return StrategyEvaluator(starting_capital=100000.0)

    @pytest.fixture
    def sample_trades(self):
        from src.model.strategy_evaluator import TradeRecord

        trades = []
        base_time = datetime.now()

        for i in range(10):
            pnl = 100.0 if i % 3 != 0 else -50.0
            trades.append(TradeRecord(
                trade_id=f"test-{i}",
                symbol="TQQQ",
                entry_time=base_time + timedelta(hours=i),
                exit_time=base_time + timedelta(hours=i+1),
                entry_price=50.0,
                exit_price=51.0 if pnl > 0 else 49.5,
                quantity=100,
                side="long",
                pnl=pnl,
                holding_period_minutes=60
            ))

        return trades

    def test_evaluator_creation(self, evaluator):
        assert evaluator.starting_capital == 100000.0

    def test_add_trade(self, evaluator):
        from src.model.strategy_evaluator import TradeRecord

        trade = TradeRecord(
            trade_id="test-1",
            symbol="TQQQ",
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=50.0,
            exit_price=51.0,
            quantity=100,
            side="long",
            pnl=100.0
        )

        evaluator.add_trade(trade)
        assert len(evaluator._trades) == 1

    def test_evaluate_basic(self, evaluator, sample_trades):
        evaluator.add_trades(sample_trades)
        metrics = evaluator.evaluate()

        assert metrics.total_trades == 10
        assert metrics.winning_trades == 7  # 7 winners, 3 losers
        assert metrics.losing_trades == 3
        assert metrics.win_rate == pytest.approx(0.7, rel=0.01)

    def test_generate_report(self, evaluator, sample_trades):
        evaluator.add_trades(sample_trades)
        report = evaluator.generate_report()

        assert "STRATEGY PERFORMANCE REPORT" in report
        assert "Win Rate" in report
        assert "Total P&L" in report


class TestDynamicRiskManager:
    """Test DynamicRiskManager class"""

    @pytest.fixture
    def risk_manager(self):
        from src.model.strategy_evaluator import DynamicRiskManager
        return DynamicRiskManager()

    def test_calculate_stops_long(self, risk_manager):
        stop_loss, take_profit = risk_manager.calculate_dynamic_stops(
            entry_price=50.0,
            current_price=50.0,
            atr=1.0,
            side="long",
            pnl_pct=0.0
        )

        assert stop_loss < 50.0
        assert take_profit > 50.0

    def test_trailing_stop_long(self, risk_manager):
        stop_loss1, _ = risk_manager.calculate_dynamic_stops(
            entry_price=50.0,
            current_price=51.0,
            atr=1.0,
            side="long",
            pnl_pct=0.02  # 2% profit
        )

        stop_loss2, _ = risk_manager.calculate_dynamic_stops(
            entry_price=50.0,
            current_price=51.0,
            atr=1.0,
            side="long",
            pnl_pct=0.0  # No profit
        )

        # Trailing stop should be higher with profit
        assert stop_loss1 > stop_loss2

    def test_should_exit_stop_loss(self, risk_manager):
        should_exit, reason = risk_manager.should_exit(
            current_price=48.0,
            stop_loss=49.0,
            take_profit=55.0,
            side="long"
        )

        assert should_exit is True
        assert reason == "stop_loss"

    def test_should_exit_take_profit(self, risk_manager):
        should_exit, reason = risk_manager.should_exit(
            current_price=56.0,
            stop_loss=49.0,
            take_profit=55.0,
            side="long"
        )

        assert should_exit is True
        assert reason == "take_profit"


class TestRewardFunction:
    """Test RewardFunction class"""

    @pytest.fixture
    def reward_func(self):
        from src.model.strategy_evaluator import RewardFunction
        return RewardFunction()

    def test_trade_reward_positive(self, reward_func):
        reward = reward_func.calculate_trade_reward(
            pnl_pct=0.02,  # 2% profit
            holding_period_minutes=60,
            slippage_pct=0.001
        )

        assert reward > 0

    def test_trade_reward_negative(self, reward_func):
        reward = reward_func.calculate_trade_reward(
            pnl_pct=-0.02,  # 2% loss
            holding_period_minutes=60,
            slippage_pct=0.001
        )

        assert reward < 0

    def test_episode_reward(self, reward_func):
        reward = reward_func.calculate_episode_reward(
            total_return_pct=0.05,
            sharpe_ratio=2.5,
            max_drawdown_pct=0.02,
            win_rate=0.7
        )

        assert reward > 0

    def test_step_reward_correct_long(self, reward_func):
        reward = reward_func.calculate_step_reward(
            action="long",
            price_change_pct=0.01,  # Price went up
            position_pnl_pct=0.01,
            signal_confidence=0.8
        )

        assert reward > 0

    def test_step_reward_wrong_long(self, reward_func):
        reward = reward_func.calculate_step_reward(
            action="long",
            price_change_pct=-0.01,  # Price went down
            position_pnl_pct=-0.01,
            signal_confidence=0.8
        )

        assert reward < 0
