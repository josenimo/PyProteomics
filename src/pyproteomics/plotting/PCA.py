def PCA_adata(adata, 
            color:str=None, 
            group_colors:dict=None,
            symbol:str=None, 
            hoverwith:list=["sampleid"],
            choose_PCs:list=[1,2],
            multi_scatter:bool=False, how_many_PCs:int=4,
            scatter_3d:bool=False,
            save_path:str=None,
            return_fig:bool=False,
            ):

    if adata.uns['pca'] is None:
        sc.pp.pca(adata, svd_solver='arpack')
        print("PCA was not found in adata.uns['pca']. It was computed now.")
    
    df = pd.DataFrame(data=adata.obsm['X_pca'], 
                        columns=[f'PC{i+1}' for i in range(adata.obsm['X_pca'].shape[1])], 
                        index=adata.obs.index)
    df = pd.concat([df, adata.obs], axis=1)

    if multi_scatter and scatter_3d:
        print("Please choose between multi_scatter and scatter_3d. Not both.")
        return None
    
    elif multi_scatter:
        features = [ f'PC{i+1}' for i in range(how_many_PCs)]
        components = df[features].values
        labels = {str(i): f"PC {i+1} ({var:.1f}%)" 
                for i, var in enumerate(adata.uns['pca']['variance_ratio']*100)}
        fig = px.scatter_matrix(
            components, labels=labels, dimensions=range(how_many_PCs), color=df[color], symbol=df[symbol])
        fig.update_traces(diagonal_visible=False,
                        marker={'size': 18, 'opacity': 0.8})
        dimension = how_many_PCs*500
        fig.update_layout(height=dimension,width=dimension,
                        font=dict(size=20, color='black'),)
        if save_path is not None:
            fig.write_image(save_path, engine='kaleido') 
        if return_fig:
            return fig

    elif scatter_3d:
        features = [ f'PC{i+1}' for i in range(3)]
        components = df[features].values
        fig = px.scatter_3d(
            components, 
            x=0, y=1, z=2, 
            color=df[color],
            symbol=df[symbol],
            labels={'0': f'PC1  {adata.uns["pca"]["variance_ratio"][0]*100:.2f}%', 
                    '1': f'PC2  {adata.uns["pca"]["variance_ratio"][1]*100:.2f}%', 
                    '2': f'PC3  {adata.uns["pca"]["variance_ratio"][2]*100:.2f}%'},
            )
        fig.update_layout(width=1000, height=1000)
        if save_path is not None:
            fig.write_html(save_path)
        if return_fig:
            return fig
        
    
    else:
        fig = px.scatter(df, x=f'PC{choose_PCs[0]}', y=f'PC{choose_PCs[1]}', 
                        color=color, 
                        symbol=symbol, 
                        hover_data=hoverwith,
                        labels={
                            f'PC{choose_PCs[0]}': 
                            f'PC{choose_PCs[0]} ({adata.uns["pca"]["variance_ratio"][choose_PCs[0]-1]*100:.2f}%)',
                            f'PC{choose_PCs[1]}': 
                            f'PC{choose_PCs[1]} ({adata.uns["pca"]["variance_ratio"][choose_PCs[1]-1]*100:.2f}%)'
                        },
                        color_discrete_map=group_colors
                        )
        fig.update_layout(
            title=dict(text=f"PCA of samples by {color} and {symbol}", font=dict(size=24), 
                        automargin=False, yref='paper'),
            font=dict( size=15, color='black'),
            width=1500,
            height=1000,
            # template='plotly_white'
            )
        fig.update_traces(
            marker={'size': 15, 'opacity': 0.8})
        
        if save_path is not None:
            fig.write_image(save_path, engine='kaleido') 
        if return_fig:
            return fig
    
    if not return_fig:
        fig.show()