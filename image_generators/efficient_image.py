import matplotlib.pyplot as plt

# Define EfficientNet-B0 architecture details (simplified for clarity)
architecture = [
    # Layer, Type, Filters, Kernel Size, Stride, Output Shape
    ("Stem", "Conv", 32, "3x3", 2, "112x112x32"),
    ("MBConv1_1", "MBConv", 16, "3x3", 1, "112x112x16"),
    ("MBConv2_1", "MBConv", 24, "3x3", 2, "56x56x24"),
    ("MBConv2_2", "MBConv", 24, "3x3", 1, "56x56x24"),
    ("MBConv3_1", "MBConv", 40, "5x5", 2, "28x28x40"),
    ("MBConv3_2", "MBConv", 40, "5x5", 1, "28x28x40"),
    ("MBConv4_1", "MBConv", 80, "3x3", 2, "14x14x80"),
    ("MBConv4_2", "MBConv", 80, "3x3", 1, "14x14x80"),
    ("MBConv4_3", "MBConv", 80, "3x3", 1, "14x14x80"),
    ("MBConv5_1", "MBConv", 112, "5x5", 1, "14x14x112"),
    ("MBConv5_2", "MBConv", 112, "5x5", 1, "14x14x112"),
    ("MBConv5_3", "MBConv", 112, "5x5", 1, "14x14x112"),
    ("MBConv6_1", "MBConv", 192, "5x5", 2, "7x7x192"),
    ("MBConv6_2", "MBConv", 192, "5x5", 1, "7x7x192"),
    ("MBConv6_3", "MBConv", 192, "5x5", 1, "7x7x192"),
    ("MBConv6_4", "MBConv", 192, "5x5", 1, "7x7x192"),
    ("MBConv7_1", "MBConv", 320, "3x3", 1, "7x7x320"),
    ("Head_Conv", "Conv", 1280, "1x1", 1, "7x7x1280"),
    ("Head_Pool", "AdaptiveAvgPool", "-", "1x1", "-", "1x1x1280"),
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
table.set_fontsize(9)  # Smaller font for more rows
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
plt.title("EfficientNet-B0 Architecture (Tan & Le, 2019)", fontsize=14, pad=20)

# Save with 600 DPI
plt.savefig("../images/efficientnetb0_architecture_table.png", dpi=600, bbox_inches='tight', format='png')
plt.close()

print("Table saved as 'efficientnetb0_architecture_table.png' with 600 DPI resolution.")