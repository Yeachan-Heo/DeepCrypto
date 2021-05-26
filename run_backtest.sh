backtest \
	--data_format sqlite3 \
	--data_path /home/ych/Storage/binance/binance.db \
	--table_name BTCUSDT_1H \
	--columns tohlcv \
	--strategy crossover.strategy \
	--output_path sample_export \
	--strategy_name crossover \
	--config_path config.json 
