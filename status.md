# Polymarket Arbitrage Bot -- Status

*Updated 2026-02-16 11:15:41*

## How the Pipeline Works

The bot runs in a continuous loop. Each iteration is called a **cycle**:

1. **Fetch markets** -- pulls all active markets from the Gamma REST API.
   Binary markets (YES/NO) and negRisk events (multi-outcome) are separated.
2. **Scan for arbitrage** -- four independent scanners run on the fetched markets:
   - **binary_rebalance**: buy arb when YES ask + NO ask < $1.00, or sell arb when YES bid + NO bid > $1.00.
   - **negrisk_rebalance**: buy arb when sum of all YES asks < $1.00, or sell arb when sum of YES bids > $1.00.
   - **latency_arb**: 15-minute crypto markets (BTC/ETH/SOL up/down) reprice slower than spot exchanges.
     The bot compares Polymarket odds to live spot momentum and buys or sells when the market lags.
   - **spike_lag**: during breaking news one market reprices instantly while sibling markets in the same
     event lag by 5-60 seconds. The bot builds a multi-leg negRisk basket on the lagging outcomes.
3. **Score and rank** -- every opportunity gets a composite score (0-1) from five weighted factors:
   profit magnitude, fill probability, capital efficiency, urgency, and competition.
   Opportunities are sorted best-first.
4. **Size** -- half-Kelly criterion determines how many sets to trade given current bankroll and edge.
   *(Skipped in DRY-RUN and SCAN-ONLY modes.)*
5. **Safety checks** -- price freshness, orderbook depth, and gas cost are verified. If any check fails
   the opportunity is skipped. A circuit breaker halts the bot on excessive losses.
   *(Skipped in DRY-RUN and SCAN-ONLY modes.)*
6. **Execute** -- FAK (fill-and-kill) orders are sent for each leg. Partial fills are unwound.
   *(Skipped in DRY-RUN and SCAN-ONLY modes.)*
7. **Record** -- P&L is updated, the trade is appended to the NDJSON ledger, and this status file is rewritten.
   *(Skipped in DRY-RUN and SCAN-ONLY modes.)*
8. **Sleep** -- the bot waits for the remaining scan interval before starting the next cycle.

## Field Reference

### Current State

| Field | Meaning |
|-------|---------|
| Mode | DRY-RUN = public APIs only, no wallet. SCAN-ONLY = detect but don't trade. PAPER = simulated fills. LIVE = real orders. |
| Uptime | Wall-clock time since the bot process started. |
| Cycle | How many full fetch-scan-execute loops have completed. |
| Markets scanned | Number of individual binary markets fetched this cycle (negRisk markets counted individually). |
| Opportunities (this cycle) | Arbitrage opportunities that passed minimum profit and ROI filters this cycle. |
| Opportunities (session) | Cumulative count across all cycles since startup. |
| Trades executed | (Trading modes only) How many opportunities were actually sent to the exchange. |
| Net P&L | (Trading modes only) Realized profit/loss across all trades this session, after fees and gas. |
| Current exposure | (Trading modes only) Total capital currently locked in open positions awaiting resolution. |

### Opportunities Table

| Column | Meaning |
|--------|---------|
| Type | Which scanner found it: binary_rebalance, negrisk_rebalance, latency_arb, or spike_lag. |
| Event | Market question for single-market arbs; event title for multi-market baskets. |
| Profit | Net expected profit in USD after subtracting gas cost and the 2% resolution fee. |
| ROI | Return on invested capital as a percentage (net_profit / required_capital * 100). |
| Score | Composite score (0-1). Weights: 25% profit, 25% fill probability, 20% capital efficiency, 20% urgency, 10% competition. |
| Legs | Number of separate orders required (2 for binary, N for negRisk, 1 for latency, N for spike). |
| Capital | USDC needed to execute all legs at the quoted prices and sizes. |

### Recent Cycles Table

| Column | Meaning |
|--------|---------|
| Cycle | Cycle number. |
| Time | Wall-clock time when the cycle completed. |
| Markets | Markets scanned that cycle. |
| Opps | Opportunities found that cycle. |
| Best Type | Scanner type of the highest-profit opportunity. |
| Best ROI | ROI of the highest-profit opportunity. |
| Best Profit | Dollar profit of the highest-profit opportunity. |
| Best Event | Event title of the highest-profit opportunity. |

---

## Current State

| Field                        | Value                                    |
|------------------------------|------------------------------------------|
| Mode                         | DRY-RUN (public APIs only, no execution) |
| Uptime                       | 9m 56s                                   |
| Cycle                        | 311                                      |
| Markets scanned              | 2,145                                    |
| Opportunities (this cycle)   | 100                                      |
| Opportunities (session)      | 30719                                    |
| Executable lane (this cycle) | 81                                       |
| Research lane (this cycle)   | 19                                       |
| Executable lane profit       | $6231.51                                 |
| Research lane profit         | $432.95                                  |

## Opportunities This Cycle

| #   | Type            | Event                                              | Profit   | ROI     | Score | Legs | Capital  |
|-----|-----------------|----------------------------------------------------|----------|---------|-------|------|----------|
| 1   | correlation_arb | MegaETH market cap (FDV) >$2B one day after lau... | $184.18  | 41.48%  | 0.66  | 2    | $444.00  |
| 2   | correlation_arb | Israel x Iran ceasefire broken by March 31, 2026?  | $143.46  | 42.45%  | 0.65  | 2    | $337.98  |
| 3   | correlation_arb | Will xAI release a dLLM by June 30?                | $197.23  | 85.18%  | 0.65  | 2    | $231.55  |
| 4   | correlation_arb | Will xAI release a dLLM by June 30?                | $175.79  | 69.49%  | 0.65  | 2    | $252.99  |
| 5   | correlation_arb | Will MetaMask launch a token by June 30?           | $168.53  | 58.72%  | 0.65  | 2    | $286.99  |
| 6   | correlation_arb | Will MetaMask launch a token by June 30?           | $235.04  | 47.06%  | 0.65  | 2    | $499.49  |
| 7   | correlation_arb | Will the US confirm that aliens exist before 2027? | $492.52  | 49.25%  | 0.65  | 2    | $1000.00 |
| 8   | correlation_arb | OpenAI $1t+ IPO before 2027?                       | $150.31  | 49.25%  | 0.65  | 2    | $305.21  |
| 9   | correlation_arb | Will the U.S. invade Iran before 2027?             | $138.45  | 40.84%  | 0.65  | 2    | $339.01  |
| 10  | correlation_arb | Will the U.S. invade Iran before 2027?             | $138.45  | 40.84%  | 0.65  | 2    | $339.01  |
| 11  | correlation_arb | Will JD Vance win the 2028 US Presidential Elec... | $1136.74 | 113.67% | 0.65  | 2    | $1000.00 |
| 12  | correlation_arb | StandX FDV above $800M one day after launch?       | $156.02  | 31.06%  | 0.65  | 2    | $502.36  |
| 13  | correlation_arb | EdgeX FDV above $1B one day after launch?          | $305.47  | 30.55%  | 0.65  | 2    | $1000.00 |
| 14  | correlation_arb | StandX FDV above $800M one day after launch?       | $129.69  | 24.53%  | 0.65  | 2    | $528.70  |
| 15  | correlation_arb | Will Trump resign before 2027?                     | $219.50  | 21.95%  | 0.65  | 2    | $1000.00 |
| 16  | correlation_arb | Will Trump resign before 2027?                     | $98.89   | 9.89%   | 0.65  | 2    | $1000.00 |
| 17  | correlation_arb | Will the U.S. invade Iran before 2027?             | $95.48   | 25.00%  | 0.65  | 2    | $381.98  |
| 18  | correlation_arb | Will xAI release a dLLM by June 30?                | $90.03   | 26.58%  | 0.65  | 2    | $338.75  |
| 19  | correlation_arb | Will MetaMask launch a token by June 30?           | $88.13   | 13.63%  | 0.64  | 2    | $646.40  |
| 20  | correlation_arb | OpenAI $1t+ IPO before 2027?                       | $423.62  | 222.57% | 0.64  | 2    | $190.33  |
| 21  | correlation_arb | MegaETH market cap (FDV) >$2B one day after lau... | $75.25   | 7.53%   | 0.64  | 2    | $1000.00 |
| 22  | correlation_arb | StandX FDV above $800M one day after launch?       | $70.43   | 11.98%  | 0.64  | 2    | $587.95  |
| 23  | correlation_arb | Will the US confirm that aliens exist before 2027? | $41.83   | 19.04%  | 0.62  | 2    | $219.69  |
| 24  | correlation_arb | Israel x Iran ceasefire broken by March 31, 2026?  | $36.83   | 7.52%   | 0.62  | 2    | $489.52  |
| 25  | correlation_arb | MegaETH market cap (FDV) >$2B one day after lau... | $30.91   | 3.09%   | 0.61  | 2    | $1000.00 |
| 26  | correlation_arb | Trump impeached by end of 2026?                    | $30.57   | 14.93%  | 0.61  | 2    | $204.69  |
| 27  | correlation_arb | Will the US confirm that aliens exist before 2027? | $30.69   | 4.16%   | 0.61  | 2    | $736.99  |
| 28  | correlation_arb | Foreign intervention in Gaza by March 31?          | $23.09   | 5.04%   | 0.61  | 2    | $458.34  |
| 29  | correlation_arb | Will Mamdani raise the minimum wage to $30 befo... | $19.98   | 4.16%   | 0.60  | 2    | $479.91  |
| 30  | correlation_arb | Will Russia test a nuclear weapon by March 31 2... | $17.35   | 6.04%   | 0.60  | 2    | $287.26  |
| 31  | correlation_arb | Ukraine signs peace deal with Russia by March 31?  | $13.69   | 2.04%   | 0.59  | 2    | $671.62  |
| 32  | correlation_arb | Trump impeached by end of 2026?                    | $9.40    | 4.16%   | 0.58  | 2    | $225.87  |
| 33  | correlation_arb | Will Russia rejoin the G7 before 2027?             | $48.18   | 31.74%  | 0.58  | 2    | $151.80  |
| 34  | correlation_arb | Will Russia test a nuclear weapon by March 31 2... | $8.21    | 2.77%   | 0.58  | 2    | $296.40  |
| 35  | correlation_arb | Will Russia rejoin the G7 before 2027?             | $72.18   | 56.48%  | 0.57  | 2    | $127.80  |
| 36  | correlation_arb | Will OpenAI launch a new consumer hardware prod... | $48.68   | 38.11%  | 0.56  | 2    | $127.74  |
| 37  | correlation_arb | Will Russia rejoin the G7 before 2027?             | $86.18   | 75.73%  | 0.56  | 2    | $113.80  |
| 38  | correlation_arb | Will OpenAI launch a new consumer hardware prod... | $61.38   | 53.36%  | 0.56  | 2    | $115.03  |
| 39  | correlation_arb | Will GTA 6 cost $100+?                             | $69.30   | 89.73%  | 0.55  | 2    | $77.23   |
| 40  | correlation_arb | Will OpenAI launch a new consumer hardware prod... | $115.72  | 190.67% | 0.54  | 2    | $60.69   |
| 41  | correlation_arb | Will OpenAI’s market cap be less than $500B at ... | $58.48   | 63.92%  | 0.53  | 2    | $91.50   |
| 42  | correlation_arb | Will USD.AI launch a token by March 31?            | $87.47   | 185.68% | 0.51  | 2    | $47.11   |
| 43  | correlation_arb | Will any presidential candidate win outright in... | $91.00   | 138.96% | 0.51  | 2    | $65.49   |
| 44  | correlation_arb | Will Russia test a nuclear weapon by March 31 2... | $29.65   | 31.04%  | 0.51  | 2    | $95.51   |
| 45  | correlation_arb | Will Russia capture Kostyantynivka by March 31?    | $20.01   | 19.03%  | 0.51  | 2    | $105.15  |
| 46  | correlation_arb | Will Mamdani raise the minimum wage to $30 befo... | $46.06   | 56.23%  | 0.51  | 2    | $81.92   |
| 47  | correlation_arb | Will OpenAI’s market cap be less than $500B at ... | $46.55   | 64.65%  | 0.50  | 2    | $72.00   |
| 48  | correlation_arb | Israel x Iran ceasefire broken by March 31, 2026?  | $28.08   | 39.06%  | 0.49  | 2    | $71.90   |
| 49  | correlation_arb | Will the U.S. test a nuclear weapon by March 31... | $57.94   | 222.52% | 0.49  | 2    | $26.04   |
| 50  | correlation_arb | Will the U.S. test a nuclear weapon by March 31... | $32.07   | 61.78%  | 0.47  | 2    | $51.91   |
| 51  | correlation_arb | Will the next UK election be called by June 30,... | $19.79   | 27.37%  | 0.47  | 2    | $72.32   |
| 52  | correlation_arb | Will the U.S. test a nuclear weapon by March 31... | $26.02   | 44.90%  | 0.47  | 2    | $57.96   |
| 53  | correlation_arb | Will Trump's approval rating hit 40% in 2026?      | $56.68   | 170.22% | 0.47  | 2    | $33.30   |
| 54  | correlation_arb | Will Israel launch a major ground offensive in ... | $3.33    | 2.87%   | 0.47  | 2    | $116.15  |
| 55  | correlation_arb | Will Trump deport 750,000 or more people in 2025?  | $13.60   | 19.88%  | 0.46  | 2    | $68.39   |
| 56  | correlation_arb | Will StandX launch a token by March 31?            | $36.45   | 170.20% | 0.46  | 2    | $21.42   |
| 57  | correlation_arb | Will Trump meet with Putin by March 31, 2026?      | $3.40    | 3.08%   | 0.46  | 2    | $110.35  |
| 58  | correlation_arb | Will USD.AI launch a token by March 31?            | $22.06   | 47.02%  | 0.46  | 2    | $46.92   |
| 59  | correlation_arb | Will Elon cut the budget by at least 5% in 2025?   | $8.02    | 174.78% | 0.46  | 2    | $4.59    |
| 60  | correlation_arb | Trump impeached by end of 2026?                    | $43.80   | 155.14% | 0.46  | 2    | $28.23   |
| 61  | correlation_arb | Will a US ally get a nuke before 2027?             | $20.70   | 40.81%  | 0.45  | 2    | $50.73   |
| 62  | correlation_arb | Will Trump deport 750,000 or more people in 2025?  | $6.22    | 8.20%   | 0.44  | 2    | $75.77   |
| 63  | correlation_arb | Will StandX launch a token by March 31?            | $18.62   | 47.45%  | 0.44  | 2    | $39.25   |
| 64  | correlation_arb | Will Trump deport less than 250,000?               | $11.70   | 22.97%  | 0.44  | 2    | $50.93   |
| 65  | correlation_arb | Will a US ally get a nuke before 2027?             | $8.56    | 13.61%  | 0.44  | 2    | $62.88   |
| 66  | correlation_arb | Bitcoin more valuable than any company before 2... | $10.79   | 21.92%  | 0.43  | 2    | $49.24   |
| 67  | correlation_arb | Will Trump create a tariff dividend by March 31?   | $1.12    | 1.00%   | 0.43  | 2    | $112.62  |
| 68  | correlation_arb | Will Trump meet with Putin by March 31, 2026?      | $1.12    | 1.00%   | 0.43  | 2    | $112.62  |
| 69  | correlation_arb | Will StandX launch a token by March 31?            | $15.98   | 222.36% | 0.43  | 2    | $7.19    |
| 70  | correlation_arb | Will Trump deport less than 250,000?               | $6.06    | 10.71%  | 0.43  | 2    | $56.56   |
| 71  | correlation_arb | Will Trump deport less than 250,000?               | $11.08   | 46.35%  | 0.42  | 2    | $23.91   |
| 72  | correlation_arb | Will Trump deport 750,000 or more people in 2025?  | $10.34   | 41.98%  | 0.41  | 2    | $24.64   |
| 73  | correlation_arb | Will Trump resign before 2027?                     | $10.83   | 44.86%  | 0.41  | 2    | $24.15   |
| 74  | correlation_arb | Will xAI release a dLLM by June 30?                | $13.96   | 116.67% | 0.41  | 2    | $11.96   |
| 75  | correlation_arb | Trump impeached by end of 2026?                    | $9.08    | 35.07%  | 0.41  | 2    | $25.90   |
| 76  | correlation_arb | MegaETH market cap (FDV) >$2B one day after lau... | $7.39    | 112.07% | 0.40  | 2    | $6.59    |
| 77  | correlation_arb | OpenAI $1t+ IPO before 2027?                       | $13.89   | 149.83% | 0.40  | 2    | $9.27    |
| 78  | correlation_arb | EdgeX FDV above $1B one day after launch?          | $9.69    | 225.36% | 0.39  | 2    | $4.30    |
| 79  | correlation_arb | Will OpenAI launch a new consumer hardware prod... | $3.46    | 216.11% | 0.39  | 2    | $1.60    |
| 80  | correlation_arb | Will Trump's approval rating hit 40% in 2026?      | $5.34    | 21.18%  | 0.39  | 2    | $25.20   |
| 81  | correlation_arb | Will OpenAI’s market cap be less than $500B at ... | $6.94    | 42.76%  | 0.39  | 2    | $16.23   |
| 82  | correlation_arb | OpenAI $1t+ IPO before 2027?                       | $9.01    | 179.65% | 0.39  | 2    | $5.01    |
| 83  | correlation_arb | Kraken IPO by March 31, 2026?                      | $5.78    | 33.24%  | 0.38  | 2    | $17.38   |
| 84  | correlation_arb | Will Trump's approval rating hit 40% in 2026?      | $3.48    | 11.06%  | 0.38  | 2    | $31.50   |
| 85  | correlation_arb | Bitcoin more valuable than any company before 2... | $6.38    | 132.84% | 0.38  | 2    | $4.80    |
| 86  | correlation_arb | Will Trump deport less than 250,000?               | $3.69    | 93.49%  | 0.37  | 2    | $3.95    |
| 87  | correlation_arb | Will OpenAI’s market cap be less than $500B at ... | $3.62    | 27.10%  | 0.37  | 2    | $13.37   |
| 88  | correlation_arb | Will Trump deport less than 250,000?               | $3.06    | 77.62%  | 0.37  | 2    | $3.95    |
| 89  | correlation_arb | Will StandX launch a token by March 31?            | $2.81    | 124.79% | 0.36  | 2    | $2.25    |
| 90  | correlation_arb | Will the U.S. test a nuclear weapon by March 31... | $2.85    | 113.11% | 0.36  | 2    | $2.52    |
| 91  | correlation_arb | Will a US ally get a nuke before 2027?             | $3.39    | 44.72%  | 0.36  | 2    | $7.59    |
| 92  | correlation_arb | Kraken IPO closing market cap above $16B?          | $1.61    | 7.45%   | 0.35  | 2    | $21.56   |
| 93  | correlation_arb | Will Trump deport less than 250,000?               | $2.00    | 50.74%  | 0.35  | 2    | $3.95    |
| 94  | correlation_arb | MegaETH market cap (FDV) >$2B one day after lau... | $1.54    | 41.38%  | 0.34  | 2    | $3.71    |
| 95  | correlation_arb | Will Trump's approval rating hit 40% in 2026?      | $0.78    | 5.16%   | 0.32  | 2    | $15.20   |
| 96  | correlation_arb | Will a US ally get a nuke before 2027?             | $0.98    | 40.57%  | 0.32  | 2    | $2.42    |
| 97  | correlation_arb | Will Trump resign before 2027?                     | $0.71    | 41.91%  | 0.30  | 2    | $1.68    |
| 98  | maker_rebalance | Will Lee Zeldin leave the Trump administration ... | $1.11    | 1.25%   | 0.25  | 2    | $88.23   |
| 99  | maker_rebalance | U.S. agrees to a new trade deal with "South Kor... | $0.41    | 0.96%   | 0.23  | 2    | $42.75   |
| 100 | maker_rebalance | Cobie mindshare all time high by March 31?         | $0.48    | 0.92%   | 0.22  | 2    | $51.70   |

## Recent Cycles

| Cycle | Time     | Markets | Opps | Best Type       | Best ROI | Best Profit | Best Event                               |
|-------|----------|---------|------|-----------------|----------|-------------|------------------------------------------|
| 311   | 11:15:41 | 2,145   | 100  | correlation_arb | 225.36%  | $1136.74    | Will JD Vance win the 2028 US Preside... |
| 310   | 11:15:40 | 2,145   | 100  | correlation_arb | 225.36%  | $1136.74    | Will JD Vance win the 2028 US Preside... |
| 309   | 11:15:39 | 2,145   | 100  | correlation_arb | 225.36%  | $1136.74    | Will JD Vance win the 2028 US Preside... |
| 308   | 11:15:38 | 2,145   | 100  | correlation_arb | 225.36%  | $1136.74    | Will JD Vance win the 2028 US Preside... |
| 307   | 11:15:37 | 2,145   | 100  | correlation_arb | 225.36%  | $1136.74    | Will JD Vance win the 2028 US Preside... |
| 306   | 11:15:36 | 2,145   | 100  | correlation_arb | 225.36%  | $1136.74    | Will JD Vance win the 2028 US Preside... |
| 305   | 11:15:35 | 2,145   | 100  | correlation_arb | 225.36%  | $1136.74    | Will JD Vance win the 2028 US Preside... |
| 304   | 11:15:35 | 2,145   | 100  | correlation_arb | 225.36%  | $1136.74    | Will JD Vance win the 2028 US Preside... |
| 303   | 11:15:33 | 2,145   | 100  | correlation_arb | 225.36%  | $1136.74    | Will JD Vance win the 2028 US Preside... |
| 302   | 11:15:33 | 2,145   | 100  | correlation_arb | 225.36%  | $1136.74    | Will JD Vance win the 2028 US Preside... |
| 301   | 11:15:31 | 2,145   | 100  | correlation_arb | 225.42%  | $1136.74    | Will JD Vance win the 2028 US Preside... |
| 300   | 11:15:30 | 2,145   | 100  | correlation_arb | 225.42%  | $1136.74    | Will JD Vance win the 2028 US Preside... |
| 299   | 11:15:29 | 2,145   | 100  | correlation_arb | 225.42%  | $1136.74    | Will JD Vance win the 2028 US Preside... |
| 298   | 11:15:28 | 2,145   | 100  | correlation_arb | 225.42%  | $1136.74    | Will JD Vance win the 2028 US Preside... |
| 297   | 11:15:27 | 2,145   | 100  | correlation_arb | 225.42%  | $1136.74    | Will JD Vance win the 2028 US Preside... |
| 296   | 11:15:26 | 2,145   | 100  | correlation_arb | 225.42%  | $1136.74    | Will JD Vance win the 2028 US Preside... |
| 295   | 11:15:25 | 2,145   | 100  | correlation_arb | 225.42%  | $1136.74    | Will JD Vance win the 2028 US Preside... |
| 294   | 11:15:24 | 2,145   | 100  | correlation_arb | 224.23%  | $1136.69    | Will JD Vance win the 2028 US Preside... |
| 293   | 11:15:23 | 2,145   | 100  | correlation_arb | 224.23%  | $1136.69    | Will JD Vance win the 2028 US Preside... |
| 292   | 11:15:23 | 2,145   | 100  | correlation_arb | 224.23%  | $1136.69    | Will JD Vance win the 2028 US Preside... |

