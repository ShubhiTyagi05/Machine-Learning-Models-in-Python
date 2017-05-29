import pandas as pd
import pandas_profiling

pandas_profiling.ProfileReport(train_data).to_file(os.path.join(output_location, "train.html"))
