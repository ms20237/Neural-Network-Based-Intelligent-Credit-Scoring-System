import argparse
import os
import pandas as pd
import glob


def init():
    parser = argparse.ArgumentParser(description="Convert 'Default of Credit Card Clients' dataset to unified feature format.")
    parser.add_argument('--dataset1', 
                        type=str, 
                        required=True,    
                        help="Path to dataset1 CSV (e.g., default_of_credit_card_clients.csv)")

    parser.add_argument('--dataset2', 
                        type=str, 
                        required=True,    
                        help="Path to dataset2 CSV (e.g., default_of_credit_card_clients.csv)")

    parser.add_argument('--dataset3', 
                        type=str, 
                        required=True,    
                        help="Path to dataset3 CSV (e.g., default_of_credit_card_clients.csv)")

    parser.add_argument('--output_path', 
                        type=str, 
                        default="./src/merged_dataset.csv",
                        help="Output path for converted CSV")

    return parser.parse_args()


def run(file1: str, 
        file2: str, 
        file3: str, 
        output_path: str):
    """
        Merge 3 CSV files with same headers vertically
    """
    try:
        # Read the CSV files
        print("Reading CSV files...")
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        df3 = pd.read_csv(file3)
        
        # Display info about each file
        print(f"\nFile 1 ({os.path.basename(file1)}): {len(df1)} rows")
        print(f"File 2 ({os.path.basename(file2)}): {len(df2)} rows")
        print(f"File 3 ({os.path.basename(file3)}): {len(df3)} rows")
        
        # Verify headers are the same
        headers1 = list(df1.columns)
        headers2 = list(df2.columns)
        headers3 = list(df3.columns)
        
        if headers1 == headers2 == headers3:
            print(f"\n✓ All files have the same headers: {headers1}")
        else:
            print("\n⚠ Warning: Headers are not identical!")
            print(f"File 1 headers: {headers1}")
            print(f"File 2 headers: {headers2}")
            print(f"File 3 headers: {headers3}")
        
        # Merge the dataframes (stack vertically)
        merged_df = pd.concat([df1, df2, df3], ignore_index=True)
        
        # Save to new CSV file
        merged_df.to_csv(output_path, index=False)
        
        print(f"\n✓ Successfully merged {len(merged_df)} total rows")
        print(f"✓ Output saved to: {output_path}")
        
    except FileNotFoundError as e:
        print(f"✗ Error: File not found - {e}")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    args = init()
    run(args.dataset1,
        args.dataset2,
        args.dataset3, 
        args.output_path)
    
    