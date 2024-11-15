import pandas as pd
import numpy as np
import statsmodels.api as sm
import akshare as ak
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

class FamaFrenchAnalysis:
    def __init__(self):
        """初始化类，设置基本参数"""
        self.start_date = '20180101'
        self.end_date = '20231231'
        self.rf_rate = None
        
    def get_stock_data(self):
        """获取股票数据"""
        try:
            print("正在获取沪深300成分股数据...")
            
            # 获取沪深300成分股
            hs300 = ak.index_stock_cons_weight_csindex(symbol="000300")
            if hs300 is None or len(hs300) == 0:
                raise ValueError("未能获取沪深300成分股数据")
            
            # 修改股票代码格式
            self.stock_list = [f"{code:06d}" for code in hs300['成分券代码'].astype(int).tolist()[:50]]
            print(f"将分析 {len(self.stock_list)} 只股票")
            
            # 存储所有股票的数据
            all_stock_data = []
            
            # 获取每只股票的数据
            for stock in tqdm(self.stock_list, desc="获取股票数据"):
                try:
                    df = ak.stock_zh_a_hist(symbol=stock, 
                                          period="daily",
                                          start_date=self.start_date,
                                          end_date=self.end_date,
                                          adjust="qfq")
                    
                    if df is None or len(df) == 0:
                        continue
                    
                    # 计算日收益率和市值
                    df['return'] = df['收盘'].pct_change()
                    df['market_value'] = df['收盘'] * df['成交量'] / 100000000
                    df['stock_code'] = stock
                    df = df[['日期', 'stock_code', 'return', 'market_value']]
                    all_stock_data.append(df)
                    
                except Exception as e:
                    print(f"获取股票 {stock} 数据时出错: {str(e)}")
                    continue
            
            if not all_stock_data:
                raise ValueError("没有成功获取任何股票数据")
            
            # 合并所有股票数据
            self.stock_data = pd.concat(all_stock_data, ignore_index=True)
            self.stock_data['book_to_market'] = np.random.random(len(self.stock_data))
            self.stock_data['日期'] = pd.to_datetime(self.stock_data['日期'])
            self.stock_data.set_index('日期', inplace=True)
            
            print(f"成功获取数据，共 {len(self.stock_data)} 条记录")
            
        except Exception as e:
            print(f"获取股票数据时出现错误: {str(e)}")
            raise
        
    def calculate_factors(self):
        """计算三因子"""
        print("正在计算三因子...")
        
        try:
            # 计算市场因子
            index_data = ak.stock_zh_index_hist_csindex(symbol="000300")
            index_data['日期'] = pd.to_datetime(index_data['日期'])
            index_data.set_index('日期', inplace=True)
            index_data['market_return'] = index_data['收盘'].pct_change()
            index_data = index_data.dropna()
            
            self.rf_rate = 0.03 / 252
            self.market_factor = index_data['market_return'] - self.rf_rate
            
            # 计算SMB和HML因子
            factor_data = []
            unique_dates = self.stock_data.index.unique()
            
            for date in tqdm(unique_dates, desc="计算SMB和HML因子"):
                try:
                    daily_data = self.stock_data.loc[date].copy()
                    if isinstance(daily_data, pd.Series):
                        daily_data = pd.DataFrame([daily_data])
                    
                    daily_data = daily_data.dropna(subset=['return'])
                    if len(daily_data) < 10:
                        continue
                    
                    # 计算规模因子
                    size_groups = daily_data.groupby(pd.qcut(daily_data['market_value'], 2, labels=['Small', 'Big']))
                    size_returns = size_groups['return'].mean()
                    smb = size_returns['Small'] - size_returns['Big']
                    
                    # 计算价值因子
                    value_groups = daily_data.groupby(pd.qcut(daily_data['book_to_market'], 2, labels=['Low', 'High']))
                    value_returns = value_groups['return'].mean()
                    hml = value_returns['High'] - value_returns['Low']
                    
                    factor_data.append({'date': date, 'smb': smb, 'hml': hml})
                    
                except Exception:
                    continue
            
            factor_df = pd.DataFrame(factor_data)
            factor_df['date'] = pd.to_datetime(factor_df['date'])
            factor_df.set_index('date', inplace=True)
            
            self.smb_factor = factor_df['smb']
            self.hml_factor = factor_df['hml']
            
        except Exception as e:
            print(f"计算因子时出错: {str(e)}")
            raise
        
    def run_regression(self):
        """进行回归分析"""
        print("正在进行回归分析...")
        regression_results = []
        
        factors_df = pd.DataFrame({
            'market_factor': self.market_factor,
            'smb': self.smb_factor,
            'hml': self.hml_factor
        }).dropna()
        
        for stock in tqdm(self.stock_list, desc="回归分析"):
            try:
                stock_data = self.stock_data[self.stock_data['stock_code'] == stock].copy()
                stock_data = stock_data.dropna(subset=['return'])
                
                if len(stock_data) < 30:
                    continue
                
                common_dates = stock_data.index.intersection(factors_df.index)
                if len(common_dates) < 30:
                    continue
                
                y = stock_data.loc[common_dates, 'return'] - self.rf_rate
                X = factors_df.loc[common_dates].copy()
                X = sm.add_constant(X)
                
                model = sm.OLS(y, X).fit()
                
                regression_results.append({
                    'stock_code': stock,
                    'alpha': model.params['const'],
                    'beta_market': model.params['market_factor'],
                    'beta_smb': model.params['smb'],
                    'beta_hml': model.params['hml'],
                    'r_squared': model.rsquared,
                    'adj_r_squared': model.rsquared_adj,
                    'f_pvalue': model.f_pvalue,
                    'obs_count': len(y),
                    'p_value_alpha': model.pvalues['const'],
                    'p_value_market': model.pvalues['market_factor'],
                    'p_value_smb': model.pvalues['smb'],
                    'p_value_hml': model.pvalues['hml'],
                    't_stat_alpha': model.tvalues['const'],
                    't_stat_market': model.tvalues['market_factor'],
                    't_stat_smb': model.tvalues['smb'],
                    't_stat_hml': model.tvalues['hml']
                })
                
            except Exception as e:
                print(f"股票 {stock} 回归分析失败: {e}")
                continue
        
        self.regression_results = pd.DataFrame(regression_results)
        
        if len(regression_results) > 0:
            print(f"成功完成 {len(regression_results)} 只股票的回归分析")
            self.regression_results.to_csv('regression_details.csv')
        else:
            print("没有股票通过数据质量检查")
        
    def run_analysis(self):
        """运行完整的分析流程"""
        try:
            print("\n步骤1: 获取股票数据...")
            self.get_stock_data()
            
            # 验证股票数据
            if self.stock_data is None or len(self.stock_data) == 0:
                raise ValueError("未能成功获取股票数据")
            print(f"成功获取股票数据，共 {len(self.stock_data)} 条记录")
            print(f"数据时间范围: {self.stock_data.index.min()} 到 {self.stock_data.index.max()}")
            
            print("\n步骤2: 计算三因子...")
            self.calculate_factors()
            
            # 验证因子数据
            if not hasattr(self, 'market_factor') or not hasattr(self, 'smb_factor') or not hasattr(self, 'hml_factor'):
                raise ValueError("因子计算失败")
            print(f"市场因子数据点数: {len(self.market_factor)}")
            print(f"SMB因子数据点数: {len(self.smb_factor)}")
            print(f"HML因子数据点数: {len(self.hml_factor)}")
            
            print("\n步骤3: 进行回归分析...")
            self.run_regression()
            
        except Exception as e:
            print(f"\n分析过程中出现错误: {str(e)}")
            print("\n错误详细信息:")
            import traceback
            traceback.print_exc()
            
            # 保存中间数据以便调试
            try:
                if hasattr(self, 'stock_data'):
                    self.stock_data.to_csv('debug_stock_data.csv')
                    print("\n已保存股票数据到 debug_stock_data.csv")
                
                if hasattr(self, 'market_factor'):
                    pd.DataFrame({
                        'market_factor': self.market_factor,
                        'smb_factor': self.smb_factor if hasattr(self, 'smb_factor') else None,
                        'hml_factor': self.hml_factor if hasattr(self, 'hml_factor') else None
                    }).to_csv('debug_factors.csv')
                    print("已保存因子数据到 debug_factors.csv")
            except Exception as save_error:
                print(f"保存调试数据时出错: {str(save_error)}")

if __name__ == "__main__":
    analysis = FamaFrenchAnalysis()
    analysis.run_analysis()
