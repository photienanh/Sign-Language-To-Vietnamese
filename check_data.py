import numpy as np
import os
import sys
from tabulate import tabulate
from tqdm import tqdm


np.set_printoptions(threshold=sys.maxsize)

def load_and_print_data(data_path):
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                data = np.load(file_path)
                print(f"Data from {file_path}:")
                print(data)
                print("\n")

def print_directory_info(data_path):
    subdirs = next(os.walk(data_path))[1]
    print(f"Number of subfolders in {data_path}: {len(subdirs)}")
    print(f"Subfolder: {subdirs}")

def count_npy_files(folder_path):
    count = 0
    for _, _, files in os.walk(folder_path):
        count += sum(1 for file in files if file.endswith('.npy'))
    return count

def analyze_directory(data_path):
    if not os.path.exists(data_path):
        print(f"{data_path} not found.")
        return

    # Thu thập thông tin
    data = []
    total_folders = 0
    total_files = 0
    
    # Lấy danh sách thư mục gốc (các hành động)
    action_folders = next(os.walk(data_path))[1]
    
    for action in sorted(action_folders):
        action_path = os.path.join(data_path, action)
        sequences = next(os.walk(action_path))[1]  # Các thư mục sequence
        npy_count = count_npy_files(action_path)
        
        total_folders += len(sequences)
        total_files += npy_count
        
        # Thêm vào bảng
        data.append([
            action,
            len(sequences),
            npy_count,
            f"{npy_count/(len(sequences) or 1):.1f}",
            "✓" if npy_count > 0 else "✗"
        ])

    # In bảng thống kê
    headers = ["Action", "Sequences", "Num of file .npy", "Mean of file/seq", "Status"]
    print("\nDetail:")
    print(tabulate(data, headers=headers, tablefmt="grid"))
    
    # In tổng quan
    print("\nOverview:")
    print(f"- Total actions: {len(action_folders)}")
    print(f"- Total sequence-folders: {total_folders}")
    print(f"- Total .npy files: {total_files}")
    print(f"- Mean of files/actions: {total_files/len(action_folders):.1f}")

def validate_npy_file(file_path):
    try:
        data = np.load(file_path)
        # Kiểm tra kích thước (phải là 126 features cho mỗi frame)
        if data.shape != (126,):
            return False, f"Shape not valid: {data.shape} (need 126,)"
        # Kiểm tra giá trị
        if np.isnan(data).any():
            return False, "Contains NaN value"
        if np.isinf(data).any():
            return False, "Contains Inf value"
        return True, "Valid"
    except Exception as e:
        return False, str(e)
    
def check_dataset_integrity(data_path):
    if not os.path.exists(data_path):
        print(f"{data_path} not found")
        return

    problems = []
    stats = {
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0
    }

    # Lấy danh sách các hành động
    actions = sorted(next(os.walk(data_path))[1])
    
    for action in tqdm(actions, desc="Checking data"):
        action_path = os.path.join(data_path, action)
        sequences = sorted(next(os.walk(action_path))[1])
        
        for seq in sequences:
            seq_path = os.path.join(action_path, seq)
            files = [f for f in os.listdir(seq_path) if f.endswith('.npy')]
            
            for file in files:
                stats['total_files'] += 1
                file_path = os.path.join(seq_path, file)
                is_valid, message = validate_npy_file(file_path)
                
                if is_valid:
                    stats['valid_files'] += 1
                else:
                    stats['invalid_files'] += 1
                    problems.append([
                        action,
                        seq,
                        file,
                        message
                    ])

    # In kết quả
    print("\nResult:")
    print(f"Total files: {stats['total_files']}")
    print(f"Valid files: {stats['valid_files']}")
    print(f"Invalid files: {stats['invalid_files']}")
    
    if problems:
        print("\nList of corrupted files:")
        headers = ["Action", "Sequence", "File", "Error"]
        print(tabulate(problems, headers=headers, tablefmt="grid"))
        
        # Thống kê lỗi theo hành động
        action_stats = {}
        for p in problems:
            action = p[0]
            if action not in action_stats:
                action_stats[action] = 0
            action_stats[action] += 1
        
        print("\nError statistics by action:")
        action_problems = [[action, count] for action, count in action_stats.items()]
        print(tabulate(action_problems, ["Action", "Number of corrupted files"], tablefmt="grid"))

def find_and_fix_sequence_issues(data_path, required_sequences=60):
    print("\nChecking the number of sequences for each action")
    
    issues = []
    for action in os.listdir(data_path):
        action_path = os.path.join(data_path, action)
        if not os.path.isdir(action_path):
            continue
            
        sequences = sorted([int(seq) for seq in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, seq))])
        if not sequences:
            continue
            
        # Kiểm tra số lượng sequence
        if len(sequences) != required_sequences:
            issues.append([
                action,
                len(sequences),
                sequences[-1] if sequences else 'N/A',
                "Incorrect number of sequences"
            ])
            
        # Kiểm tra thứ tự sequence
        for i, seq in enumerate(sequences):
            if seq >= required_sequences:
                issues.append([
                    action,
                    seq,
                    i,
                    "Sequence index exceeds the limit"
                ])
    
    if issues:
        print("\nList of sequence issues:")
        headers = ["Action", "Num sequence/Index", "Last Sequence/Position", "Issue"]
        print(tabulate(issues, headers=headers, tablefmt="grid"))
        
        # Hỏi người dùng có muốn sửa không
        fix = input("\nDo you want to fix these issues? (y/n): ")
        if fix.lower() == 'y':
            for action in os.listdir(data_path):
                action_path = os.path.join(data_path, action)
                if not os.path.isdir(action_path):
                    continue
                    
                sequences = sorted([int(seq) for seq in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, seq))])
                if not sequences:
                    continue
                    
                # Xóa các sequence thừa
                for seq in sequences:
                    if seq >= required_sequences:
                        seq_path = os.path.join(action_path, str(seq))
                        try:
                            import shutil
                            shutil.rmtree(seq_path)
                            print(f"Deleted excess sequence: {seq_path}")
                        except Exception as e:
                            print(f"Error deleting {seq_path}: {str(e)}")
            
            print("\nIssues have been fixed!")
    else:
        print("No issues found with the number of sequences.")

def main():
    data_path = 'Data'
    analyze_directory(data_path)
    find_and_fix_sequence_issues(data_path)
    print("\nStart checking data integrity...")
    check_dataset_integrity(data_path)
if __name__ == "__main__":
    main()