import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define VGG-19 architecture details
architecture = [
    # Layer, Type, Filters, Kernel Size, Stride, Padding, Output Shape
    ("Conv1_1", "Conv", 64, "3x3", 1, 1, "224x224x64"),
    ("Conv1_2", "Conv", 64, "3x3", 1, 1, "224x224x64"),
    ("Pool1", "MaxPool", "-", "2x2", 2, 0, "112x112x64"),
    ("Conv2_1", "Conv", 128, "3x3", 1, 1, "112x112x128"),
    ("Conv2_2", "Conv", 128, "3x3", 1, 1, "112x112x128"),
    ("Pool2", "MaxPool", "-", "2x2", 2, 0, "56x56x128"),
    ("Conv3_1", "Conv", 256, "3x3", 1, 1, "56x56x256"),
    ("Conv3_2", "Conv", 256, "3x3", 1, 1, "56x56x256"),
    ("Conv3_3", "Conv", 256, "3x3", 1, 1, "56x56x256"),
    ("Conv3_4", "Conv", 256, "3x3", 1, 1, "56x56x256"),
    ("Pool3", "MaxPool", "-", "2x2", 2, 0, "28x28x256"),
    ("Conv4_1", "Conv", 512, "3x3", 1, 1, "28x28x512"),
    ("Conv4_2", "Conv", 512, "3x3", 1, 1, "28x28x512"),
    ("Conv4_3", "Conv", 512, "3x3", 1, 1, "28x28x512"),
    ("Conv4_4", "Conv", 512, "3x3", 1, 1, "28x28x512"),
    ("Pool4", "MaxPool", "-", "2x2", 2, 0, "14x14x512"),
    ("Conv5_1", "Conv", 512, "3x3", 1, 1, "14x14x512"),
    ("Conv5_2", "Conv", 512, "3x3", 1, 1, "14x14x512"),
    ("Conv5_3", "Conv", 512, "3x3", 1, 1, "14x14x512"),
    ("Conv5_4", "Conv", 512, "3x3", 1, 1, "14x14x512"),
    ("Pool5", "MaxPool", "-", "2x2", 2, 0, "7x7x512"),
    ("FC6", "Fully Connected", 4096, "-", "-", "-", "4096"),
    ("FC7", "Fully Connected", 4096, "-", "-", "-", "4096"),
    ("FC8", "Fully Connected", 1000, "-", "-", "-", "1000"),
]

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))  # Slightly larger for readability
ax.axis('off')  # Hide axes

# Define table data
col_labels = ["Layer", "Type", "Filters", "Kernel Size", "Stride", "Padding", "Output Shape"]
table_data = architecture

# Create the table
table = ax.table(cellText=table_data,
                 colLabels=col_labels,
                 cellLoc='center',
                 loc='center',
                 colColours=['#f0f0f0'] * len(col_labels),  # Light gray header
                 bbox=[0, 0, 1, 1])  # Full figure size

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)  # Scale columns and rows for better spacing

# Adjust cell properties
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('black')
    cell.set_linewidth(0.5)
    if row == 0:  # Header row
        cell.set_text_props(weight='bold', color='black')
        cell.set_facecolor('#d3d3d3')  # Slightly darker gray for header
    else:
        cell.set_text_props(color='black')
        cell.set_facecolor('#ffffff')  # White background for data rows

# Add title
plt.title("VGG-19 Architecture (Simonyan & Zisserman, 2014)", fontsize=14, pad=20)

# Save the figure with 600 DPI
plt.savefig("images/vgg19_architecture_table.png", dpi=600, bbox_inches='tight', format='png')
plt.close()  # Close to free memory

print("Table saved as 'vgg19_architecture_table.png' with 600 DPI resolution.")