import pandas as pd
import plotly.express as px

def plot_boxplots_plotly(adata, 
                        x_axis="Phenotype_1", 
                        y_axis:str="n_proteins", 
                        hover_data:list=["Phenotype_2"], 
                        color_column="Phenotype_1",
                        return_fig=False,
                        save_path:str=None,
                        save_df_path:str=None,
                        **kwargs):

    adata_copy = adata.copy()

    df = pd.DataFrame(index=adata_copy.obs.index, data=adata_copy.obs.values, columns=adata_copy.obs_keys())

    fig = px.box(df, x=x_axis, y=y_axis, 
        points='all', hover_data=hover_data, 
        color=color_column, width=1000, height=800,
        color_discrete_sequence=px.colors.qualitative.G10,
        **kwargs
        )

    fig.update_layout(
        title=dict(text="Proteins per sample", 
                font=dict(size=30), 
                automargin=True, 
                yref='paper'),
        font=dict( size=18, color='black'),
        paper_bgcolor='rgb(255, 255, 255)',
        plot_bgcolor='rgb(255, 255, 255)',
        showlegend=True,   
    )

    if save_df_path is not None:
        df.to_csv(save_df_path)
    if save_path is not None:
        # plt.savefig(save_path, format="png")
        fig.write_image(save_path, engine='kaleido') 


    if return_fig:
        return fig
    else:
        fig.show()