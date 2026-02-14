# Polymarket Arbitrage Bot -- Status

*Updated 2026-02-14 18:42:10*

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
| Event | Human-readable event title from Polymarket (e.g. "Will BTC be above 100k?"). |
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

| Field                      | Value                                    |
|----------------------------|------------------------------------------|
| Mode                       | DRY-RUN (public APIs only, no execution) |
| Uptime                     | 1m 37s                                   |
| Cycle                      | 65                                       |
| Markets scanned            | 14,520                                   |
| Opportunities (this cycle) | 165                                      |
| Opportunities (session)    | 9865                                     |

## Opportunities This Cycle

| #   | Type              | Event                                              | Profit | ROI      | Score | Legs | Capital |
|-----|-------------------|----------------------------------------------------|--------|----------|-------|------|---------|
| 1   | maker_rebalance   | Will Paris St-Germain finish in the top 4 of th... | $26.52 | 930.56%  | 0.54  | 2    | $2.85   |
| 2   | maker_rebalance   | Will Paris St-Germain finish in the top 4 of th... | $10.94 | 1404.25% | 0.54  | 2    | $0.78   |
| 3   | maker_rebalance   | Will Bayern Munich finish in the top 4 of the B... | $7.34  | 1426.09% | 0.53  | 2    | $0.51   |
| 4   | maker_rebalance   | Will Paris St-Germain finish in the top 4 of th... | $20.16 | 715.74%  | 0.52  | 2    | $2.82   |
| 5   | maker_rebalance   | Will Donald Trump visit the United Kingdom in 2... | $64.76 | 117.73%  | 0.51  | 2    | $55.01  |
| 6   | maker_rebalance   | Will Bernie endorse James Talarico for TX-Sen b... | $52.13 | 97.03%   | 0.51  | 2    | $53.72  |
| 7   | maker_rebalance   | Will Trump reduce the deficit before 2027?         | $45.40 | 53.09%   | 0.50  | 2    | $85.51  |
| 8   | maker_rebalance   | Will Paris St-Germain finish in the top 4 of th... | $5.86  | 1009.05% | 0.49  | 2    | $0.58   |
| 9   | maker_rebalance   | Will Barcelona finish in the top 4 of the La Li... | $32.53 | 26.01%   | 0.49  | 2    | $125.10 |
| 10  | maker_rebalance   | Will Trump endorse John Cornyn for TX-Sen by No... | $27.02 | 205.95%  | 0.49  | 2    | $13.12  |
| 11  | maker_rebalance   | Will Bayern Munich finish in the top 4 of the B... | $7.80  | 721.02%  | 0.49  | 2    | $1.08   |
| 12  | maker_rebalance   | Will Espanyol be relegated from La Liga after t... | $17.84 | 269.32%  | 0.49  | 2    | $6.62   |
| 13  | maker_rebalance   | Will Paris St-Germain finish in the top 4 of th... | $10.13 | 478.55%  | 0.48  | 2    | $2.12   |
| 14  | maker_rebalance   | Will Cremonese be relegated from Serie A after ... | $10.65 | 389.05%  | 0.48  | 2    | $2.74   |
| 15  | maker_rebalance   | Will Bayern Munich finish in the top 4 of the B... | $14.64 | 107.34%  | 0.47  | 2    | $13.64  |
| 16  | maker_rebalance   | Will Bayern Munich finish in the top 4 of the B... | $8.71  | 310.74%  | 0.47  | 2    | $2.80   |
| 17  | maker_rebalance   | Will Bayern Munich finish in the top 4 of the B... | $12.72 | 108.29%  | 0.47  | 2    | $11.75  |
| 18  | maker_rebalance   | Will Barcelona finish in the top 4 of the La Li... | $8.28  | 332.48%  | 0.47  | 2    | $2.49   |
| 19  | maker_rebalance   | Will Conrad Kramer leave OpenAI by December 31,... | $13.47 | 122.49%  | 0.46  | 2    | $11.00  |
| 20  | maker_rebalance   | Will Barcelona finish in the top 4 of the La Li... | $12.55 | 74.74%   | 0.46  | 2    | $16.79  |
| 21  | maker_rebalance   | Will Consensys IPO by June 30 2026?                | $12.22 | 99.79%   | 0.46  | 2    | $12.25  |
| 22  | maker_rebalance   | Will Paris St-Germain finish in the top 4 of th... | $8.57  | 262.17%  | 0.46  | 2    | $3.27   |
| 23  | maker_rebalance   | Will Paris St-Germain finish in the top 4 of th... | $11.97 | 68.82%   | 0.46  | 2    | $17.40  |
| 24  | maker_rebalance   | Will Barcelona finish in the top 4 of the La Li... | $8.64  | 138.61%  | 0.46  | 2    | $6.23   |
| 25  | maker_rebalance   | Will Trump endorse John Cornyn for TX-Sen by No... | $10.47 | 74.81%   | 0.45  | 2    | $14.00  |
| 26  | maker_rebalance   | Will Barcelona finish in the top 4 of the La Li... | $9.19  | 85.62%   | 0.45  | 2    | $10.73  |
| 27  | maker_rebalance   | Will Espanyol be relegated from La Liga after t... | $6.32  | 283.05%  | 0.45  | 2    | $2.23   |
| 28  | maker_rebalance   | Will Cremonese be relegated from Serie A after ... | $7.53  | 179.38%  | 0.45  | 2    | $4.20   |
| 29  | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $9.32  | 20.93%   | 0.45  | 2    | $44.55  |
| 30  | maker_rebalance   | Will Paris St-Germain finish in the top 4 of th... | $9.32  | 18.49%   | 0.45  | 2    | $50.38  |
| 31  | maker_rebalance   | Will Ari Weinstein leave OpenAI by December 31,... | $8.14  | 28.59%   | 0.45  | 2    | $28.49  |
| 32  | maker_rebalance   | Will Bayern Munich finish in the top 4 of the B... | $3.87  | 384.87%  | 0.44  | 2    | $1.01   |
| 33  | maker_rebalance   | Will Bayern Munich finish in the top 4 of the B... | $3.90  | 355.47%  | 0.44  | 2    | $1.10   |
| 34  | maker_rebalance   | Will Paris St-Germain finish in the top 4 of th... | $5.92  | 117.26%  | 0.44  | 2    | $5.05   |
| 35  | maker_rebalance   | Will Cremonese be relegated from Serie A after ... | $6.72  | 37.88%   | 0.44  | 2    | $17.75  |
| 36  | maker_rebalance   | Will Cremonese be relegated from Serie A after ... | $6.24  | 55.32%   | 0.44  | 2    | $11.27  |
| 37  | maker_rebalance   | Will Paris St-Germain finish in the top 4 of th... | $6.69  | 16.23%   | 0.44  | 2    | $41.25  |
| 38  | maker_rebalance   | Will Espanyol be relegated from La Liga after t... | $6.26  | 39.83%   | 0.44  | 2    | $15.71  |
| 39  | maker_rebalance   | Will Bayern Munich finish in the top 4 of the B... | $6.09  | 48.28%   | 0.44  | 2    | $12.61  |
| 40  | maker_rebalance   | Will Napoli finish in the top 4 in the 2025-26 ... | $5.09  | 86.13%   | 0.44  | 2    | $5.91   |
| 41  | maker_rebalance   | Will the Atlanta Hawks win more than 47.5 regul... | $4.70  | 58.67%   | 0.43  | 2    | $8.01   |
| 42  | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $5.07  | 20.88%   | 0.43  | 2    | $24.30  |
| 43  | maker_rebalance   | Will Paris St-Germain finish in the top 4 of th... | $4.80  | 47.68%   | 0.43  | 2    | $10.06  |
| 44  | maker_rebalance   | Will the Atlanta Hawks win more than 47.5 regul... | $4.37  | 28.78%   | 0.43  | 2    | $15.20  |
| 45  | maker_rebalance   | Will Union Berlin be relegated from the Bundesl... | $4.52  | 19.40%   | 0.43  | 2    | $23.30  |
| 46  | maker_rebalance   | Will Napoli finish in the top 4 in the 2025-26 ... | $3.81  | 72.25%   | 0.43  | 2    | $5.27   |
| 47  | maker_rebalance   | Will Liverpool be relegated from the English Pr... | $4.14  | 20.61%   | 0.42  | 2    | $20.07  |
| 48  | maker_rebalance   | Will Bayern Munich finish in the top 4 of the B... | $3.50  | 91.48%   | 0.42  | 2    | $3.83   |
| 49  | maker_rebalance   | Will Cremonese be relegated from Serie A after ... | $3.60  | 50.40%   | 0.42  | 2    | $7.15   |
| 50  | maker_rebalance   | Will Union Berlin be relegated from the Bundesl... | $3.82  | 8.37%    | 0.42  | 2    | $45.64  |
| 51  | maker_rebalance   | Will Bayern Munich finish in the top 4 of the B... | $3.22  | 90.20%   | 0.42  | 2    | $3.57   |
| 52  | maker_rebalance   | Will Liverpool be relegated from the English Pr... | $3.29  | 10.30%   | 0.42  | 2    | $31.96  |
| 53  | maker_rebalance   | Will the US recognize Palestine before 2027?       | $3.21  | 4.22%    | 0.42  | 2    | $76.14  |
| 54  | maker_rebalance   | Will Trump endorse John Cornyn for TX-Sen by No... | $2.93  | 121.64%  | 0.42  | 2    | $2.41   |
| 55  | maker_rebalance   | Will Union Berlin be relegated from the Bundesl... | $2.88  | 48.05%   | 0.41  | 2    | $6.00   |
| 56  | maker_rebalance   | Will Paris St-Germain finish in the top 4 of th... | $2.76  | 71.26%   | 0.41  | 2    | $3.87   |
| 57  | maker_rebalance   | Will Barcelona finish in the top 4 of the La Li... | $2.60  | 88.21%   | 0.41  | 2    | $2.95   |
| 58  | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $2.86  | 3.25%    | 0.41  | 2    | $88.16  |
| 59  | maker_rebalance   | Will Consensys IPO by June 30 2026?                | $2.79  | 6.46%    | 0.41  | 2    | $43.15  |
| 60  | maker_rebalance   | Will the Atlanta Hawks win more than 47.5 regul... | $2.38  | 43.75%   | 0.41  | 2    | $5.44   |
| 61  | maker_rebalance   | Will Consensys IPO by June 30 2026?                | $2.65  | 19.32%   | 0.41  | 2    | $13.73  |
| 62  | maker_rebalance   | Will Napoli finish in the top 4 in the 2025-26 ... | $2.60  | 11.91%   | 0.41  | 2    | $21.85  |
| 63  | maker_rebalance   | Will the Atlanta Hawks win more than 47.5 regul... | $2.45  | 14.37%   | 0.41  | 2    | $17.07  |
| 64  | maker_rebalance   | U.S. forces in Gaza before 2027?                   | $2.54  | 17.89%   | 0.41  | 2    | $14.18  |
| 65  | maker_rebalance   | Will Lorient be relegated from Ligue 1 after th... | $2.32  | 46.53%   | 0.41  | 2    | $5.00   |
| 66  | maker_rebalance   | Will Lorient be relegated from Ligue 1 after th... | $2.26  | 59.97%   | 0.41  | 2    | $3.78   |
| 67  | maker_rebalance   | Will Cremonese be relegated from Serie A after ... | $2.20  | 65.33%   | 0.41  | 2    | $3.37   |
| 68  | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $2.12  | 28.83%   | 0.41  | 2    | $7.36   |
| 69  | maker_rebalance   | Will Trump cut corporate taxes before 2027?        | $2.30  | 10.00%   | 0.41  | 2    | $23.04  |
| 70  | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $2.22  | 10.00%   | 0.40  | 2    | $22.25  |
| 71  | maker_rebalance   | Will Liverpool be relegated from the English Pr... | $2.18  | 5.89%    | 0.40  | 2    | $36.96  |
| 72  | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $1.97  | 33.80%   | 0.40  | 2    | $5.84   |
| 73  | maker_rebalance   | Will Paris St-Germain finish in the top 4 of th... | $1.92  | 65.22%   | 0.40  | 2    | $2.95   |
| 74  | maker_rebalance   | Will Tarcisio de Frietas qualify for Brazil's p... | $2.11  | 3.45%    | 0.40  | 2    | $61.11  |
| 75  | maker_rebalance   | Will Donald Trump visit the United Kingdom in 2... | $2.01  | 12.48%   | 0.40  | 2    | $16.09  |
| 76  | maker_rebalance   | Obama federally charged before 2027?               | $1.93  | 8.91%    | 0.40  | 2    | $21.62  |
| 77  | maker_rebalance   | Will the Atlanta Hawks win more than 47.5 regul... | $1.77  | 19.23%   | 0.40  | 2    | $9.23   |
| 78  | maker_rebalance   | Will Lorient be relegated from Ligue 1 after th... | $1.73  | 10.85%   | 0.40  | 2    | $15.97  |
| 79  | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $1.72  | 7.58%    | 0.40  | 2    | $22.75  |
| 80  | maker_rebalance   | Will Oman join the Abraham Accords before 2027?    | $1.72  | 23.69%   | 0.40  | 2    | $7.27   |
| 81  | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $1.67  | 7.57%    | 0.40  | 2    | $22.04  |
| 82  | maker_rebalance   | Will Cremonese be relegated from Serie A after ... | $1.59  | 35.99%   | 0.40  | 2    | $4.42   |
| 83  | maker_rebalance   | Will Union Berlin be relegated from the Bundesl... | $1.58  | 16.44%   | 0.39  | 2    | $9.64   |
| 84  | maker_rebalance   | Will Arsenal finish in the top 4 of the EPL 202... | $1.60  | 7.01%    | 0.39  | 2    | $22.85  |
| 85  | maker_rebalance   | Spain snap election called by June 30, 2026?       | $1.55  | 6.41%    | 0.39  | 2    | $24.17  |
| 86  | maker_rebalance   | Will Espanyol be relegated from La Liga after t... | $1.50  | 35.77%   | 0.39  | 2    | $4.20   |
| 87  | maker_rebalance   | Will Barcelona finish in the top 4 of the La Li... | $1.50  | 35.96%   | 0.39  | 2    | $4.17   |
| 88  | maker_rebalance   | Will Donald Trump visit the United Kingdom in 2... | $1.57  | 19.19%   | 0.39  | 2    | $8.20   |
| 89  | maker_rebalance   | Will the US recognize Palestine before 2027?       | $1.52  | 8.74%    | 0.39  | 2    | $17.36  |
| 90  | maker_rebalance   | Will Donald Trump visit the United Kingdom in 2... | $1.50  | 9.94%    | 0.39  | 2    | $15.13  |
| 91  | maker_rebalance   | Will Napoli finish in the top 4 in the 2025-26 ... | $1.47  | 3.10%    | 0.39  | 2    | $47.50  |
| 92  | maker_rebalance   | Will Barcelona finish in the top 4 of the La Li... | $1.38  | 5.96%    | 0.39  | 2    | $23.07  |
| 93  | maker_rebalance   | Will Arsenal finish in the top 4 of the EPL 202... | $1.38  | 1.43%    | 0.39  | 2    | $96.50  |
| 94  | maker_rebalance   | Will Liverpool be relegated from the English Pr... | $1.31  | 8.86%    | 0.39  | 2    | $14.82  |
| 95  | maker_rebalance   | Israel and Lebanon normalize relations before 2... | $1.31  | 5.27%    | 0.39  | 2    | $24.92  |
| 96  | maker_rebalance   | Will Mitch McConnell step down from the Senate ... | $1.29  | 13.68%   | 0.39  | 2    | $9.46   |
| 97  | maker_rebalance   | Will the Atlanta Hawks win more than 47.5 regul... | $1.28  | 4.18%    | 0.39  | 2    | $30.60  |
| 98  | maker_rebalance   | Will Barcelona finish in the top 4 of the La Li... | $1.23  | 18.44%   | 0.39  | 2    | $6.68   |
| 99  | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $1.25  | 4.17%    | 0.39  | 2    | $30.08  |
| 100 | maker_rebalance   | Will Bayern Munich finish in the top 4 of the B... | $1.15  | 21.61%   | 0.38  | 2    | $5.34   |
| 101 | maker_rebalance   | Will Paris St-Germain finish in the top 4 of th... | $1.16  | 13.55%   | 0.38  | 2    | $8.59   |
| 102 | maker_rebalance   | Will Bernie endorse James Talarico for TX-Sen b... | $1.14  | 15.52%   | 0.38  | 2    | $7.37   |
| 103 | maker_rebalance   | Will Donald Trump visit the United Kingdom in 2... | $1.13  | 26.66%   | 0.38  | 2    | $4.24   |
| 104 | maker_rebalance   | Will Barcelona finish in the top 4 of the La Li... | $1.09  | 8.68%    | 0.38  | 2    | $12.60  |
| 105 | maker_rebalance   | Will the US recognize Palestine before 2027?       | $1.09  | 8.68%    | 0.38  | 2    | $12.60  |
| 106 | maker_rebalance   | Obama arrested before 2027?                        | $1.08  | 4.60%    | 0.38  | 2    | $23.38  |
| 107 | maker_rebalance   | US national Ethereum reserve before 2027?          | $1.03  | 3.08%    | 0.38  | 2    | $33.53  |
| 108 | maker_rebalance   | Will the US recognize Palestine before 2027?       | $1.02  | 15.75%   | 0.38  | 2    | $6.50   |
| 109 | maker_rebalance   | Will Oman join the Abraham Accords before 2027?    | $1.02  | 5.24%    | 0.38  | 2    | $19.37  |
| 110 | maker_rebalance   | Will the Atlanta Hawks win more than 47.5 regul... | $0.97  | 11.07%   | 0.38  | 2    | $8.80   |
| 111 | maker_rebalance   | Will Liverpool be relegated from the English Pr... | $0.98  | 15.47%   | 0.38  | 2    | $6.34   |
| 112 | maker_rebalance   | Will Bayern Munich finish in the top 4 of the B... | $0.95  | 16.22%   | 0.38  | 2    | $5.88   |
| 113 | maker_rebalance   | Will Paris St-Germain finish in the top 4 of th... | $0.97  | 3.19%    | 0.38  | 2    | $30.47  |
| 114 | maker_rebalance   | Will Donald Trump visit the United Kingdom in 2... | $0.91  | 20.40%   | 0.38  | 2    | $4.46   |
| 115 | maker_rebalance   | Will Oman join the Abraham Accords before 2027?    | $0.91  | 11.05%   | 0.38  | 2    | $8.24   |
| 116 | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $0.87  | 3.07%    | 0.37  | 2    | $28.50  |
| 117 | maker_rebalance   | Will AppLovin acquire TikTok?                      | $0.88  | 4.25%    | 0.37  | 2    | $20.67  |
| 118 | maker_rebalance   | Will Liverpool be relegated from the English Pr... | $0.86  | 3.07%    | 0.37  | 2    | $28.06  |
| 119 | maker_rebalance   | Will the US recognize Palestine before 2027?       | $0.85  | 6.33%    | 0.37  | 2    | $13.43  |
| 120 | maker_rebalance   | Will Donald Trump visit the United Kingdom in 2... | $0.85  | 5.22%    | 0.37  | 2    | $16.27  |
| 121 | maker_rebalance   | Will Donald Trump visit the United Kingdom in 2... | $0.81  | 9.80%    | 0.37  | 2    | $8.31   |
| 122 | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $0.77  | 8.60%    | 0.37  | 2    | $9.00   |
| 123 | maker_rebalance   | Will Consensys IPO by June 30 2026?                | $0.76  | 9.78%    | 0.37  | 2    | $7.73   |
| 124 | maker_rebalance   | Will Italy qualify for the 2026 FIFA World Cup?    | $0.76  | 6.30%    | 0.37  | 2    | $12.02  |
| 125 | maker_rebalance   | Will Bayern Munich finish in the top 4 of the B... | $0.74  | 14.55%   | 0.37  | 2    | $5.11   |
| 126 | negrisk_rebalance | Will SpaceX's market cap be less than $1.0T at ... | $0.59  | 9.65%    | 0.37  | 8    | $6.16   |
| 127 | maker_rebalance   | Will the Atlanta Hawks win more than 47.5 regul... | $0.72  | 5.31%    | 0.37  | 2    | $13.55  |
| 128 | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $0.72  | 4.11%    | 0.37  | 2    | $17.48  |
| 129 | maker_rebalance   | Will Trump endorse John Cornyn for TX-Sen by No... | $0.67  | 7.41%    | 0.37  | 2    | $9.10   |
| 130 | maker_rebalance   | Will Liverpool be relegated from the English Pr... | $0.68  | 1.40%    | 0.37  | 2    | $48.25  |
| 131 | maker_rebalance   | Will the US recognize Palestine before 2027?       | $0.67  | 5.18%    | 0.37  | 2    | $13.02  |
| 132 | maker_rebalance   | Will Donald Trump visit the United Kingdom in 2... | $0.62  | 14.68%   | 0.36  | 2    | $4.25   |
| 133 | maker_rebalance   | Will Wrexham be promoted to the EPL?               | $0.58  | 9.68%    | 0.36  | 2    | $6.03   |
| 134 | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $0.57  | 8.50%    | 0.36  | 2    | $6.75   |
| 135 | maker_rebalance   | Will the Atlanta Hawks win more than 47.5 regul... | $0.57  | 6.24%    | 0.36  | 2    | $9.20   |
| 136 | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $0.57  | 4.07%    | 0.36  | 2    | $14.10  |
| 137 | maker_rebalance   | Will Bayern Munich finish in the top 4 of the B... | $0.57  | 4.61%    | 0.36  | 2    | $12.38  |
| 138 | maker_rebalance   | Cap on gambling loss deductions repealed before... | $0.57  | 5.14%    | 0.36  | 2    | $11.16  |
| 139 | maker_rebalance   | Will North Korea invade South Korea before 2027?   | $0.55  | 2.30%    | 0.36  | 2    | $23.90  |
| 140 | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $0.52  | 10.83%   | 0.36  | 2    | $4.84   |
| 141 | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $0.52  | 10.83%   | 0.36  | 2    | $4.84   |
| 142 | maker_rebalance   | Will Union Berlin be relegated from the Bundesl... | $0.53  | 7.69%    | 0.36  | 2    | $6.94   |
| 143 | maker_rebalance   | Will Espanyol be relegated from La Liga after t... | $0.53  | 3.01%    | 0.36  | 2    | $17.57  |
| 144 | maker_rebalance   | Will Bernie endorse James Talarico for TX-Sen b... | $0.50  | 2.09%    | 0.36  | 2    | $23.95  |
| 145 | maker_rebalance   | Will the Atlanta Hawks win more than 47.5 regul... | $0.48  | 0.98%    | 0.36  | 2    | $49.47  |
| 146 | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $0.47  | 5.10%    | 0.36  | 2    | $9.30   |
| 147 | maker_rebalance   | Will AppLovin acquire TikTok?                      | $0.47  | 1.68%    | 0.35  | 2    | $28.26  |
| 148 | maker_rebalance   | Obama divorce before 2027?                         | $0.47  | 1.97%    | 0.35  | 2    | $24.00  |
| 149 | maker_rebalance   | Will Trump's approval rating hit 40% in 2026?      | $0.47  | 1.97%    | 0.35  | 2    | $24.00  |
| 150 | maker_rebalance   | Will the US recognize Palestine before 2027?       | $0.47  | 1.97%    | 0.35  | 2    | $24.00  |
| 151 | maker_rebalance   | Will Oman join the Abraham Accords before 2027?    | $0.47  | 1.97%    | 0.35  | 2    | $24.00  |
| 152 | maker_rebalance   | Will Liverpool be relegated from the English Pr... | $0.45  | 7.52%    | 0.35  | 2    | $6.03   |
| 153 | maker_rebalance   | Will the Atlanta Hawks win more than 47.5 regul... | $0.44  | 3.09%    | 0.35  | 2    | $14.35  |
| 154 | maker_rebalance   | Will the Atlanta Hawks win more than 47.5 regul... | $0.42  | 6.82%    | 0.35  | 2    | $6.21   |
| 155 | maker_rebalance   | Will Espanyol be relegated from La Liga after t... | $0.42  | 1.77%    | 0.35  | 2    | $24.03  |
| 156 | maker_rebalance   | Will Liverpool be relegated from the English Pr... | $0.41  | 1.86%    | 0.35  | 2    | $21.93  |
| 157 | maker_rebalance   | Will Barcelona finish in the top 4 of the La Li... | $0.39  | 8.69%    | 0.35  | 2    | $4.48   |
| 158 | maker_rebalance   | Will Donald Trump visit the United Kingdom in 2... | $0.39  | 1.95%    | 0.35  | 2    | $19.86  |
| 159 | maker_rebalance   | Will Napoli finish in the top 4 in the 2025-26 ... | $0.38  | 5.47%    | 0.35  | 2    | $6.94   |
| 160 | maker_rebalance   | Will Oman join the Abraham Accords before 2027?    | $0.38  | 0.96%    | 0.35  | 2    | $39.28  |
| 161 | maker_rebalance   | Will Napoli finish in the top 4 in the 2025-26 ... | $0.37  | 5.36%    | 0.35  | 2    | $6.95   |
| 162 | maker_rebalance   | Will Napoli finish in the top 4 in the 2025-26 ... | $0.37  | 5.25%    | 0.35  | 2    | $7.08   |
| 163 | maker_rebalance   | Will Trump endorse John Cornyn for TX-Sen by No... | $0.37  | 3.98%    | 0.35  | 2    | $9.40   |
| 164 | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $0.36  | 7.17%    | 0.35  | 2    | $5.00   |
| 165 | maker_rebalance   | Will the Boston Bruins make the NHL Playoffs?      | $0.36  | 7.17%    | 0.35  | 2    | $5.00   |

## Recent Cycles

| Cycle | Time     | Markets | Opps | Best Type       | Best ROI | Best Profit | Best Event                               |
|-------|----------|---------|------|-----------------|----------|-------------|------------------------------------------|
| 65    | 18:42:10 | 14,520  | 165  | maker_rebalance | 1426.09% | $64.76      | Will Donald Trump visit the United Ki... |
| 64    | 18:42:09 | 14,520  | 165  | maker_rebalance | 1426.09% | $64.76      | Will Donald Trump visit the United Ki... |
| 63    | 18:42:08 | 14,520  | 165  | maker_rebalance | 1426.09% | $64.76      | Will Donald Trump visit the United Ki... |
| 62    | 18:42:07 | 14,520  | 165  | maker_rebalance | 1426.09% | $64.76      | Will Donald Trump visit the United Ki... |
| 61    | 18:42:06 | 14,520  | 165  | maker_rebalance | 1426.09% | $64.76      | Will Donald Trump visit the United Ki... |
| 60    | 18:42:05 | 14,520  | 165  | maker_rebalance | 1426.09% | $64.76      | Will Donald Trump visit the United Ki... |
| 59    | 18:42:04 | 14,520  | 153  | maker_rebalance | 1408.50% | $64.67      | Will Donald Trump visit the United Ki... |
| 58    | 18:42:03 | 14,520  | 152  | maker_rebalance | 1408.50% | $64.67      | Will Donald Trump visit the United Ki... |
| 57    | 18:42:02 | 14,520  | 152  | maker_rebalance | 1408.81% | $64.67      | Will Donald Trump visit the United Ki... |
| 56    | 18:42:01 | 14,520  | 152  | maker_rebalance | 1408.81% | $64.67      | Will Donald Trump visit the United Ki... |
| 55    | 18:42:00 | 14,520  | 152  | maker_rebalance | 1408.81% | $64.67      | Will Donald Trump visit the United Ki... |
| 54    | 18:41:59 | 14,520  | 152  | maker_rebalance | 1408.81% | $64.67      | Will Donald Trump visit the United Ki... |
| 53    | 18:41:58 | 14,520  | 152  | maker_rebalance | 1408.81% | $64.67      | Will Donald Trump visit the United Ki... |
| 52    | 18:41:57 | 14,520  | 152  | maker_rebalance | 1408.81% | $64.67      | Will Donald Trump visit the United Ki... |
| 51    | 18:41:56 | 14,520  | 152  | maker_rebalance | 1408.81% | $64.67      | Will Donald Trump visit the United Ki... |
| 50    | 18:41:55 | 14,520  | 152  | maker_rebalance | 1408.81% | $64.67      | Will Donald Trump visit the United Ki... |
| 49    | 18:41:42 | 14,529  | 150  | maker_rebalance | 1408.55% | $64.67      | Will Donald Trump visit the United Ki... |
| 48    | 18:41:41 | 14,529  | 150  | maker_rebalance | 1408.55% | $64.67      | Will Donald Trump visit the United Ki... |
| 47    | 18:41:40 | 14,529  | 150  | maker_rebalance | 1408.55% | $64.67      | Will Donald Trump visit the United Ki... |
| 46    | 18:41:39 | 14,529  | 150  | maker_rebalance | 1408.55% | $64.67      | Will Donald Trump visit the United Ki... |

