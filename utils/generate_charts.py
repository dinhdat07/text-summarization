import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

# General Setup
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

def plot_general_metrics():
    # Data from Table 1 (General Set)
    methods = ['Lead-3', 'TextRank', 'FFT - BARTpho', 'LoRA - BARTpho', 'LoRA - Qwen2.5']
    rouge1 = [45.75, 36.88, 50.24, 50.17, 47.85]
    rouge2 = [21.87, 18.99, 15.27, 14.93, 20.42]
    rougel = [28.99, 24.94, 30.52, 30.40, 30.22]
    bleu = [8.79, 6.37, 2.25, 2.34, 9.19]
    bertscore = [72.69, 71.52, 66.46, 66.29, 70.34]

    x = np.arange(len(methods))
    width = 0.15

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot ROUGE and BLEU on left y-axis
    rects1 = ax1.bar(x - 2*width, rouge1, width, label='ROUGE-1', color='#4c72b0')
    rects2 = ax1.bar(x - width, rouge2, width, label='ROUGE-2', color='#55a868')
    rects3 = ax1.bar(x, rougel, width, label='ROUGE-L', color='#c44e52')
    rects4 = ax1.bar(x + width, bleu, width, label='BLEU', color='#8172b2')

    # Plot BERTScore on right y-axis (since it's a different scale/meaning though still 0-100)
    # Actually, all are 0-100 scale, so we can plot them on the same axis for simplicity
    rects5 = ax1.bar(x + 2*width, bertscore, width, label='BERTScore-F1', color='#ccb974')

    ax1.set_ylabel('Điểm số (%)', fontsize=14)
    ax1.set_title('So sánh hiệu năng các phương pháp trên tập General Set', fontsize=16, pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right', fontsize=12)
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=2)

    # Add text labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, rotation=90)
            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)

    fig.tight_layout()
    plt.savefig('paper/Images/general_metrics.png', dpi=300)
    plt.close()

def plot_long_context_metrics():
    # Data from Table 2 (Long Document Set)
    methods = ['FFT - BARTpho\n(Truncation 1024)', 'FFT - BARTpho\n(Sliding Window)', 'LoRA - Qwen2.5\n(Native 2048)']
    rouge1 = [47.79, 47.05, 45.38]
    rouge2 = [12.05, 12.49, 16.81]
    rougel = [26.94, 26.16, 27.06]
    bleu = [1.22, 0.64, 6.27]
    bertscore = [63.64, 62.98, 68.94]

    x = np.arange(len(methods))
    width = 0.15

    fig, ax1 = plt.subplots(figsize=(10, 6))

    rects1 = ax1.bar(x - 2*width, rouge1, width, label='ROUGE-1', color='#4c72b0')
    rects2 = ax1.bar(x - width, rouge2, width, label='ROUGE-2', color='#55a868')
    rects3 = ax1.bar(x, rougel, width, label='ROUGE-L', color='#c44e52')
    rects4 = ax1.bar(x + width, bleu, width, label='BLEU', color='#8172b2')
    rects5 = ax1.bar(x + 2*width, bertscore, width, label='BERTScore-F1', color='#ccb974')

    ax1.set_ylabel('Điểm số (%)', fontsize=14)
    ax1.set_title('So sánh hiệu năng trên tập Long Document Set', fontsize=16, pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=12)
    ax1.legend(loc='upper right')
    
    # Add text labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)

    fig.tight_layout()
    plt.savefig('paper/Images/long_context_metrics.png', dpi=300)
    plt.close()

def plot_radar_chart():
    labels = np.array(['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'BERTScore-F1', 'ROUGE-Lsum'])
    num_vars = len(labels)
    
    # Data from Table 1
    # Scale all metrics relative to their max value across all models to visualize "profile shape"
    # Actually, standard radar charts plot the raw or min-max scaled values. Let's normalize by max of each metric to make it readable.
    raw_data = {
        'Lead-3': [45.75, 21.87, 28.99, 8.79, 72.69, 28.98],
        'FFT - BARTpho': [50.24, 15.27, 30.52, 2.25, 66.46, 30.51],
        'LoRA - Qwen2.5': [47.85, 20.42, 30.22, 9.19, 70.34, 30.25]
    }
    
    # Normalizing data to 0-1 range for each axis
    # (val - min) / (max - min) to show relative strengths, or just raw / max?
    # Raw/max is better so 0 is origin
    max_vals = [55, 25, 35, 10, 80, 35] # Hand-picked to fit nicely
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = ['#c44e52', '#4c72b0', '#55a868']
    
    for (method, values), color in zip(raw_data.items(), colors):
        # Normalize
        norm_values = [v / m for v, m in zip(values, max_vals)]
        norm_values += norm_values[:1]
        
        ax.plot(angles, norm_values, color=color, linewidth=2, label=method)
        ax.fill(angles, norm_values, color=color, alpha=0.1)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12)
    
    # Hide radial ticks
    ax.set_yticklabels([])
    ax.grid(color='#AAAAAA')
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.title('Hồ sơ hiệu năng (Performance Profile) tương đối', y=1.08, fontsize=16)
    
    plt.tight_layout()
    plt.savefig('paper/Images/radar_chart_models.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_general_metrics()
    plot_long_context_metrics()
    plot_radar_chart()
    print("Charts generated successfully.")
