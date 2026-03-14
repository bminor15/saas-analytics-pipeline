-- daily active users with a 30-day rolling average
-- raw DAU is noisy (weekends dip, etc.) — the rolling avg shows the real trend
-- a widening gap between dau and dau_30d_avg signals a trend shift worth investigating

SELECT
    date_day,
    year_month,
    is_weekend,
    SUM(dau)                                                AS dau,
    SUM(total_events)                                       AS total_events,
    SUM(logins)                                             AS logins,
    SUM(errors)                                             AS errors,
    ROUND(AVG(SUM(dau)) OVER (
        ORDER BY date_day
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ), 0)                                                   AS dau_30d_avg
FROM mart_daily_active_users
GROUP BY date_day, year_month, is_weekend
ORDER BY date_day;
