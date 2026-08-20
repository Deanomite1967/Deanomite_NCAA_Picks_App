from io import BytesIO
from openpyxl import load_workbook

def autosize_excel(df):
    """
    Convert DataFrame to an auto-sized Excel file and return a BytesIO buffer.
    """

    # Step 1: Write DataFrame to Excel
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")

    # Step 2: Load workbook to adjust column widths
    buffer.seek(0)
    wb = load_workbook(buffer)
    ws = wb.active

    # Step 3: Auto-size each column
    for col in ws.columns:
        max_length = 0
        col_letter = col[0].column_letter

        for cell in col:
            try:
                cell_length = len(str(cell.value))
                if cell_length > max_length:
                    max_length = cell_length
            except:
                pass

        ws.column_dimensions[col_letter].width = max_length + 2  # padding

    # Step 4: Save adjusted workbook back into buffer
    final_buffer = BytesIO()
    wb.save(final_buffer)
    final_buffer.seek(0)

    return final_buffer
