# query_agent_prompt = """

#         You are analyzing a quarterly financial report for Dipped Products PLC (DIPD.N0000) and quarterly financial report for Richard Pieris PLC (REXP.N0000).
        
#         ==================== For Richard Pieris PLC (REXP.N0000) ====================

#         TASK 1: First, determine which quarter and year this report covers.
#         Look for statements like:
#         - "Quarter ended [Date]"
#         - "Three months ended [Date]"
#         - "Period ended [Date]"
#         - "Results for the quarter ended [Date]"
        
#         If the report uses fiscal year, map to calendar quarters:
#         - Q1: First 3 months of fiscal year
#         - Q2: Months 4-6 of fiscal year
#         - Q3: Months 7-9 of fiscal year
#         - Q4: Months 10-12 of fiscal year
        
#         Example: "Three months ended June 30, 2023" with March 31 fiscal year-end would be Q1 2023.
        
#         TASK 2: Extract the following financial metrics:
#         1. Revenue/Turnover (in thousands)
#         2. Cost of Goods Sold/Cost of Sales (in thousands) 
#         3. Gross Profit (in thousands)
#         4. Total Operating Expenses (in thousands)
#         5. Operating Income/Operating Profit (in thousands)
#         6. Finance Income (in thousands)
#         7. Finance Cost (in thousands)
#         8. Profit Before Tax (in thousands)
#         9. Tax Expense (in thousands)
#         10. Net Income/Profit after tax (in thousands)
#         11. Earnings Per Share (EPS)

#         IMPORTANT: Focus ONLY on the "Group unaudited 03 months" column in the Statement of Profit or Loss.
#         Use only Group/Consolidated figures, NOT company-specific statements.
#         sometimes the EPS is not in the "Group unaudited 03 months" column you'll have to calculate it manually using below guide.

#         ✅ How EPS Is Usually Calculated
#         The basic formula is:
        
#         EPS = Net Income attributable to equity shareholders / Number of Ordinary Shares
#         Net Income = Profit attributable to ordinary shareholders
#         Number of Shares = Weighted average number of ordinary shares in circulation during the period
        
#         ✅ How EPS Is Shown in These Reports
#         In the reports from Richard Pieris PLC (REXP), EPS is:
        
#         Explicitly stated in the income statement
#         Typically found in the final section of the Consolidated Statement of Profit or Loss
#         Appears right below or next to the line: “Profit attributable to ordinary shareholders”
        
#         ✅ Example from Your Report 771_1730893408597.pdf:
#         In the Consolidated Income Statement (3 months ended 30th September 2024):
        
#         Profit attributable to shareholders: 63,605
#         EPS (Rs.): 5.70
#         This means the company earned Rs. 5.70 per share during this quarter, and this figure is:
        
#         Already pre-calculated by the company
#         Auditor-approved (or unaudited Group figure) as per the requirement
        
#         ⚠️ Important for Prompting or Automation:
#         You don’t need to calculate EPS manually unless it's missing — just extract it as-is from the correct column:
        
#         Look for the line labeled “Earnings Per Share (EPS)” or “Basic EPS” in the Group Unaudited – 03 months column.
#         If EPS is missing, fallback logic is:
        
#         Try estimating EPS only if Net Income and Number of Shares are both available.
#         Otherwise, return "eps": null.
        
#         Respond with a SINGLE JSON object containing all information:
#         {{
#             "quarter": "Q1", (must be in format Q1, Q2, Q3, or Q4)
#             "year": "2023", (4-digit year)
#             "fiscal_year": "2023/24", (if mentioned in the report)
#             "revenue": 503513.0,
#             "cogs": 366084.0,
#             "gross_profit": 137429.0,
#             "operating_expenses": 64410.0,
#             "operating_income": 73035.0,
#             "finance_income": 34481.0,
#             "finance_cost": 701.0,
#             "profit_before_tax": 107052.0,
#             "tax_expense": 32807.0,
#             "net_income": 74245.0,
#             "eps": 6.65
#         }}
        
#         Use null for any values you cannot find. Return ONLY the JSON object, no additional text.

#         ======= For Dipped Products PLC (DIPD.N0000) ======
        
#         TASK 1: First, determine which quarter and year this report covers.
#         Look for statements like:
#         - "Quarter ended [Date]"
#         - "Three months ended [Date]"
#         - "Period ended [Date]"
#         - "Results for the quarter ended [Date]"
        
#         If the report uses fiscal year, map to calendar quarters:
#         - Q1: First 3 months of fiscal year
#         - Q2: Months 4-6 of fiscal year
#         - Q3: Months 7-9 of fiscal year
#         - Q4: Months 10-12 of fiscal year
        
#         Example: "Three months ended June 30, 2023" with March 31 fiscal year-end would be Q1 2023.
        
#         TASK 2: Extract the following financial metrics:
#         1. Revenue/Turnover (in thousands)
#         2. Cost of Goods Sold/Cost of Sales (in thousands) 
#         3. Gross Profit (in thousands)
#         4. Total Operating Expenses (in thousands)
#         5. Operating Income/Operating Profit (in thousands)
#         6. Finance Income (in thousands)
#         7. Finance Cost (in thousands)
#         8. Profit Before Tax (in thousands)
#         9. Tax Expense (in thousands)
#         10. Net Income/Profit after tax (in thousands)
#         11. Earnings Per Share (EPS)

#         IMPORTANT: Focus ONLY on the Consolidated Statement of Financial Position or Consolidated Income Statement sections.
#         Use only Consolidated figures, NOT company-specific statements.
#         sometimes the EPS is not in the "Group unaudited 03 months" column you'll have to calculate it manually using below guide.

#         ✅ How EPS Is Usually Calculated
#         The basic formula is:
        
#         EPS = Net Income attributable to equity shareholders / Number of Ordinary Shares
#         Net Income = Profit attributable to ordinary shareholders
#         Number of Shares = Weighted average number of ordinary shares in circulation during the period
        
#         ✅ How EPS Is Shown in These Reports
#         In the reports from Richard Pieris PLC (REXP), EPS is:
        
#         Explicitly stated in the income statement
#         Typically found in the final section of the Consolidated Statement of Profit or Loss
#         Appears right below or next to the line: “Profit attributable to ordinary shareholders”
        
#         ✅ Example from Your Report 771_1730893408597.pdf:
#         In the Consolidated Income Statement (3 months ended 30th September 2024):
        
#         Profit attributable to shareholders: 63,605
#         EPS (Rs.): 5.70
#         This means the company earned Rs. 5.70 per share during this quarter, and this figure is:
        
#         Already pre-calculated by the company
#         Auditor-approved (or unaudited Group figure) as per the requirement
        
#         ⚠️ Important for Prompting or Automation:
#         You don’t need to calculate EPS manually unless it's missing — just extract it as-is from the correct column:
        
#         Look for the line labeled “Earnings Per Share (EPS)” or “Basic EPS” in the Group Unaudited – 03 months column.
#         If EPS is missing, fallback logic is:
        
#         Try estimating EPS only if Net Income and Number of Shares are both available.
#         Otherwise, return "eps": null.
        
#         Respond with a SINGLE JSON object containing all information:
#         {{
#             "quarter": "Q3", (must be in format Q1, Q2, Q3, or Q4)
#             "year": "2023", (4-digit year)
#             "fiscal_year": "2023/24", (if mentioned in the report)
#             "revenue": 503513.0,
#             "cogs": 366084.0,
#             "gross_profit": 137429.0,
#             "operating_expenses": 64410.0,
#             "operating_income": 73035.0,
#             "finance_income": 34481.0,
#             "finance_cost": 701.0,
#             "profit_before_tax": 107052.0,
#             "tax_expense": 32807.0,
#             "net_income": 74245.0,
#             "eps": 6.65
#         }}
        
#         Use null for any values you cannot find. Return ONLY the JSON object, no additional text.
#         you have access to following tools: {}
        


# """


single_report_prompt = """
You are an AI financial analyst reviewing a single quarterly report of a Sri Lankan PLC company.
Your goal is to extract the following financial metrics from the **attached report** and return a structured JSON.

Do NOT assume or fabricate any values. Only use what is clearly visible in this single report.

Fields to extract:
- quarter (e.g., "Q3")
- year (4-digit format)
- fiscal_year (e.g., "2022/23")
- period_end_date (e.g., "2023-09-30")
- revenue
- cogs
- gross_profit
- operating_expenses
- operating_income
- finance_income
- finance_cost
- profit_before_tax
- tax_expense
- net_income
- eps

Format:
{
    "quarter": "Q2",
    "year": "2023",
    "fiscal_year": "2023/24",
    "period_end_date": "2023-09-30",
    "revenue": 345234.0,
    "cogs": 234000.0,
    "gross_profit": 111234.0,
    "operating_expenses": 56000.0,
    "operating_income": 55234.0,
    "finance_income": 4000.0,
    "finance_cost": 1200.0,
    "profit_before_tax": 58034.0,
    "tax_expense": 14000.0,
    "net_income": 44034.0,
    "eps": 6.75
}

If a value is missing, write null.
Return only the JSON. No explanations or extra commentary.
"""


