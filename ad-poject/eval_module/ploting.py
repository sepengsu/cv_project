import os
import numpy as np
import matplotlib.pyplot as plt

def plot_and_save_score_distribution(
    train_scores,
    val_scores,
    test_scores,
    threshold,
    loss_name,
    model_name,
    test_labels=None,  # ✅ 추가됨
    plot_dir="./score_plots"
):
    os.makedirs(plot_dir, exist_ok=True)

    # ✅ 1D 평탄화
    train_scores = np.ravel(train_scores)
    val_scores = np.ravel(val_scores)
    test_scores = np.ravel(test_scores)
    
    # ✅ test 분리
    if test_labels is not None:
        test_labels = np.ravel(test_labels)
        normal_scores = test_scores[test_labels == 0] # 정상 데이터
        abnormal_scores = test_scores[test_labels != 0] # 비정상 데이터
    else:
        normal_scores = test_scores
        abnormal_scores = np.array([])

    # ✅ 전체 score 기반 x축 확대 범위 (1~99%)
    all_scores = np.concatenate([train_scores, val_scores, test_scores])
    x_min, x_max = np.percentile(all_scores, [1, 99])

    plt.figure(figsize=(9, 5))

    # ✅ 정규화된 빈도 기반 (비율)
    plt.hist(train_scores, bins=50, weights=np.ones_like(train_scores) / len(train_scores), alpha=0.4, label='Train', color='green')
    plt.hist(val_scores, bins=50, weights=np.ones_like(val_scores) / len(val_scores), alpha=0.4, label='Val', color='blue')
    if len(normal_scores) > 0:
        plt.hist(normal_scores, bins=50, weights=np.ones_like(normal_scores) / len(normal_scores), alpha=0.4, label='Test-Normal', color='orange')
    if len(abnormal_scores) > 0:
        plt.hist(abnormal_scores, bins=50, weights=np.ones_like(abnormal_scores) / len(abnormal_scores), alpha=0.4, label='Test-Abnormal', color='red')

    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.4f})')

    # ✅ 확대 적용
    plt.xlim(x_min, x_max)

    plt.xlabel("Anomaly Score")
    plt.ylabel("Proportion")
    plt.title(f"{model_name} | {loss_name} - Normal vs Abnormal Score Distribution")
    plt.legend()
    plt.grid(True)

    filename = f"{model_name.replace('/', '_')}And_loss{loss_name.replace('/', '_')}_score_dist.png"
    filepath = os.path.join(plot_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return filepath

def plot_test_score_distribution(
    test_scores,
    threshold,
    loss_name,
    model_name,
    test_labels=None,
    plot_dir="./score_plots"
):
    os.makedirs(plot_dir, exist_ok=True)

    # ✅ 1D 평탄화
    test_scores = np.ravel(test_scores)
    
    # ✅ test 분리
    if test_labels is not None:
        test_labels = np.ravel(test_labels)
        normal_scores = test_scores[test_labels == 0]  # 정상
        abnormal_scores = test_scores[test_labels != 0]  # 비정상
    else:
        normal_scores = test_scores
        abnormal_scores = np.array([])

    # ✅ 확대 범위 (1~99%)
    x_min, x_max = np.percentile(test_scores, [1, 99])

    plt.figure(figsize=(8, 5))

    # ✅ test score만 시각화
    if len(normal_scores) > 0:
        plt.hist(normal_scores, bins=50, weights=np.ones_like(normal_scores) / len(normal_scores),
                 alpha=0.5, label='Test-Normal', color='orange')
    if len(abnormal_scores) > 0:
        plt.hist(abnormal_scores, bins=50, weights=np.ones_like(abnormal_scores) / len(abnormal_scores),
                 alpha=0.5, label='Test-Abnormal', color='red')

    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.4f})')

    plt.xlim(x_min, x_max)
    plt.xlabel("Anomaly Score")
    plt.ylabel("Proportion")
    plt.title(f"[Test Only] {model_name} | {loss_name}")
    plt.legend()
    plt.grid(True)

    filename = f"{model_name}_loss{loss_name}_TESTONLY_score_dist.png".replace('/', '_')
    filepath = os.path.join(plot_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return filepath