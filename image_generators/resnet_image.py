import matplotlib.pyplot as plt

# Define ResNet-50 architecture details
architecture = [
    # Layer, Type, Filters, Kernel Size, Stride, Output Shape
    ("Conv1", "Conv", 64, "7x7", 2, "112x112x64"),
    ("Pool1", "MaxPool", "-", "3x3", 2, "56x56x64"),
    ("Conv2_1", "Bottleneck", "256 (64,64,256)", "1x1,3x3,1x1", 1, "56x56x256"),
    ("Conv2_2", "Bottleneck", "256 (64,64,256)", "1x1,3x3,1x1", 1, "56x56x256"),
    ("Conv2_3", "Bottleneck", "256 (64,64,256)", "1x1,3x3,1x1", 1, "56x56x256"),
    ("Conv3_1", "Bottleneck", "512 (128,128,512)", "1x1,3x3,1x1", 2, "28x28x512"),
    ("Conv3_2", "Bottleneck", "512 (128,128,512)", "1x1,3x3,1x1", 1, "28x28x512"),
    ("Conv3_3", "Bottleneck", "512 (128,128,512)", "1x1,3x3,1x1", 1, "28x28x512"),
    ("Conv3_4", "Bottleneck", "512 (128,128,512)", "1x1,3x3,1x1", 1, "28x28x512"),
    ("Conv4_1", "Bottleneck", "1024 (256,256,1024)", "1x1,3x3,1x1", 2, "14x14x1024"),
    ("Conv4_2", "Bottleneck", "1024 (256,256,1024)", "1x1,3x3,1x1", 1, "14x14x1024"),
    ("Conv4_3", "Bottleneck", "1024 (256,256,1024)", "1x1,3x3,1x1", 1, "14x14x1024"),
    ("Conv4_4", "Bottleneck", "1024 (256,256,1024)", "1x1,3x3,1x1", 1, "14x14x1024"),
    ("Conv4_5", "Bottleneck", "1024 (256,256,1024)", "1x1,3x3,1x1", 1, "14x14x1024"),
    ("Conv4_6", "Bottleneck", "1024 (256,256,1024)", "1x1,3x3,1x1", 1, "14x14x1024"),
    ("Conv5_1", "Bottleneck", "2048 (512,512,2048)", "1x1,3x3,1x1", 2, "7x7x2048"),
    ("Conv5_2", "Bottleneck", "2048 (512,512,2048)", "1x1,3x3,1x1", 1, "7x7x2048"),
    ("Conv5_3", "Bottleneck", "2048 (512,512,2048)", "1x1,3x3,1x1", 1, "7x7x2048"),
    ("AvgPool", "AdaptiveAvgPool", "-", "1x1", "-", "1x1x2048"),
    ("FC", "Fully Connected", 1000, "-", "-", "1000"),
]

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 10))  # Adjusted size for more layers
ax.axis('off')

# Define table data
col_labels = ["Layer", "Type", "Filters", "Kernel Size", "Stride", "Output Shape"]
table_data = architecture

# Create the table
table = ax.table(cellText=table_data,
                 colLabels=col_labels,
                 cellLoc='center',
                 loc='center',
                 colColours=['#f0f0f0'] * len(col_labels),
                 bbox=[0, 0, 1, 1])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(9)  # Slightly smaller font due to more rows
table.scale(1.2, 1.5)

# Adjust cell properties
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('black')
    cell.set_linewidth(0.5)
    if row == 0:  # Header
        cell.set_text_props(weight='bold', color='black')
        cell.set_facecolor('#d3d3d3')
    else:
        cell.set_text_props(color='black')
        cell.set_facecolor('#ffffff')

# Add title
plt.title("ResNet-50 Architecture (He et al., 2015)", fontsize=14, pad=20)

# Save with 600 DPI
plt.savefig("images/resnet50_architecture_table.png", dpi=600, bbox_inches='tight', format='png')
plt.close()

print("Table saved as 'resnet50_architecture_table.png' with 600 DPI resolution.")