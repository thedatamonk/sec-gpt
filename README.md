The SEC Agent

1. **Query validator**
    - check whether the query can be answered using SEC filings only.
    - check whether the query is relevant to the SEC agent. For instance, queries like "how many colors are there in the rainbow" should be disregarded.
    - check whether the query is about a valid public company registered in the SEC database.
    - check whether the query is about a valid data duration. For instance, if the query is asking about MoM revenue of Apple for year 2029.


2. Planning Agent
The planning agent is called after the query validator agent.
    - The planning agent "thinks" and breaks down the original query into smaller steps if required.
    - Each step needs to be executed with/without a tool.
    - The results are finally aggregated.


**What kind of information is available in SEC filings - 10K (annual) and 10Q (quarterly)?**


Three main financial statements - 
1. Income statement
2. Balance sheet
3. Cashflows

**Note** The 10K includes audited financial statements while 10Q are unaudited.

📈 **Income Statement Metrics**
The income statement, also known as the statement of earnings or statement of operations, provides insights into a company's profitability over a specific period. Metrics you can extract include:

**Revenue**: The total income from the company's business activities.

**Cost of Goods Sold (COGS)**: The direct costs attributable to the production of the goods or services sold by a company.

**Gross Profit**: Revenue minus COGS.

**Operating Expenses**: Costs not directly related to production, such as selling, general, and administrative (SG&A) expenses and research and development (R&D) expenses.

**Operating Income**: Gross profit minus operating expenses.

**Net Income (or Earnings)**: The company's total profit after all expenses, interest, and taxes have been deducted.

**Earnings Per Share (EPS)**: The portion of a company's profit allocated to each share of stock.

🏛️ **Balance Sheet Metrics**
The balance sheet provides a snapshot of a company's financial position at a single point in time. It follows the fundamental accounting equation: Assets = Liabilities + Shareholders' Equity. Metrics you can extract include:

Assets:

- Current Assets: Cash, accounts receivable, and inventory.
- Long-Term Assets: Property, plant, and equipment (PP&E), and intangible assets.

Liabilities:
- Current Liabilities: Accounts payable and short-term debt.
- Long-Term Liabilities: Long-term debt and deferred tax liabilities.

Shareholders' Equity: The amount of assets remaining after all liabilities are paid, including retained earnings and common stock.


💸 **Cash Flow Statement Metrics**
The statement of cash flows tracks how much cash and cash equivalents are generated and used by a company in a given period. It's often considered a more accurate picture of a company's financial health than the income statement because it isn't affected by non-cash accounting methods. The statement is divided into three sections:

**Cash Flow from Operating Activities (CFO)**: Cash generated from a company's normal business operations.

**Cash Flow from Investing Activities (CFI)**: Cash used for or generated from investing in assets or selling assets, such as purchasing or selling PP&E.

**Cash Flow from Financing Activities (CFF)**: Cash related to debt, equity, and dividends. This includes issuing or repurchasing stock, taking on or paying off debt, and paying dividends.
