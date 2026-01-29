from typing import List


class Calculator:
    """
    Base class providing generic mathematical operations.
    No finance-specific logic (SRP).
    """

    def add(self, a: float, b: float) -> float:
        return a + b

    def subtract(self, a: float, b: float) -> float:
        return a - b

    def multiply(self, a: float, b: float) -> float:
        return a * b

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ZeroDivisionError("Division by zero is not allowed.")
        return a / b

    def power(self, base: float, exponent: float) -> float:
        return base ** exponent


class FinancialCalculator(Calculator):
    """
    Provides reusable time-value-of-money calculations.
    Still generic: no instrument-specific logic.
    """

    def present_value(self, cash_flow: float, rate: float, period: int) -> float:
        """
        PV = CF / (1 + r)^t
        """
        discount_factor = self.power(1 + rate, period)
        return self.divide(cash_flow, discount_factor)

    def future_value(self, cash_flow: float, rate: float, period: int) -> float:
        """
        FV = CF * (1 + r)^t
        """
        growth_factor = self.power(1 + rate, period)
        return self.multiply(cash_flow, growth_factor)


class Bond(FinancialCalculator):
    """
    Represents a fixed-coupon bond and computes its analytics.
    Encapsulates all bond-specific attributes.
    """

    def __init__(
        self,
        face_value: float,
        coupon_rate: float,
        yield_to_maturity: float,
        maturity_years: int,
        payments_per_year: int = 1,
    ):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.yield_to_maturity = yield_to_maturity
        self.maturity_years = maturity_years
        self.payments_per_year = payments_per_year

        # Derived attributes
        self.total_periods = maturity_years * payments_per_year
        self.periodic_coupon = (
            face_value * coupon_rate / payments_per_year
        )
        self.periodic_yield = yield_to_maturity / payments_per_year

    def _cash_flows(self) -> List[float]:
        """
        Generates the bond's cash flow schedule.
        """
        cash_flows = [self.periodic_coupon] * self.total_periods
        cash_flows[-1] += self.face_value  # principal repayment at maturity
        return cash_flows

    def price(self) -> float:
        """
        Bond Price = Sum of PV of all cash flows
        """
        price = 0.0
        for t, cf in enumerate(self._cash_flows(), start=1):
            price += self.present_value(cf, self.periodic_yield, t)
        return price

    def macaulay_duration(self) -> float:
        """
        Macaulay Duration:
        D = Σ [ t * PV(CF_t) ] / Bond Price
        Expressed in years.
        """
        bond_price = self.price()
        weighted_sum = 0.0

        for t, cf in enumerate(self._cash_flows(), start=1):
            pv_cf = self.present_value(cf, self.periodic_yield, t)
            weighted_sum += t * pv_cf

        duration_in_periods = weighted_sum / bond_price
        return duration_in_periods / self.payments_per_year

    def modified_duration(self) -> float:
        """
        Modified Duration:
        D_mod = D_mac / (1 + y / m)
        """
        mac_dur = self.macaulay_duration()
        return mac_dur / (1 + self.periodic_yield)

    def convexity(self) -> float:
        """
        Modified convexity (in years^2).

        C = (1 / P) * Σ [ CF_t * t(t+1) / (1+y)^(t+2) ]
        Adjusted for payment frequency.
        """
        bond_price = self.price()
        convexity_sum = 0.0

        for t, cf in enumerate(self._cash_flows(), start=1):
            discount = self.power(1 + self.periodic_yield, t + 2)
            convexity_sum += cf * t * (t + 1) / discount

        # Convert from periods^2 to years^2
        return convexity_sum / bond_price / (self.payments_per_year ** 2)


    def price_percentage_change(self, delta_yield: float) -> float:
        """
        Approximates percentage price change using
        duration + convexity.

        ΔP / P ≈ -D_mod * Δy + 0.5 * Conv * (Δy)^2
        """
        d_mod = self.modified_duration()
        conv = self.convexity()

        return (
            -d_mod * delta_yield
            + 0.5 * conv * (delta_yield ** 2)
        )

if __name__ == "__main__":
    bond = Bond(
        face_value=1000,
        coupon_rate=0.06,        # 5%
        yield_to_maturity=0.08,  # 6%
        maturity_years=5,
        payments_per_year=1
    )

    print(f"Bond Price: {bond.price():.2f}")
    print(f"Macaulay Duration: {bond.macaulay_duration():.4f} years")
    print(f"Modified Duration: {bond.modified_duration():.4f}")
    delta_y = 0.01  # 1% yield increase
    pct_change = bond.price_percentage_change(delta_y)
    print(f"Approximate Price Change for 1% yield increase: {pct_change * 100:.2f}%")
