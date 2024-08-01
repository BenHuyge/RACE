function backProj = astra_create_backprojection3d_cudaHR(sinogramData, proj_geom, vol_geom)
%--------------------------------------------------------------------------
% backProj = astra_create_backprojection_cudaHR(sinogramData, proj_geom, vol_geom)
% 
% Creates a CUDA-based simple backprojection. Works exactly as astra_create_backprojection_cuda
% but allows for more than 2000 projection angles.

recon_id = astra_mex_data3d('create', '-vol', vol_geom, 0);

% split up projection geometry in blocks
[proj_geoms, indices] = splitup(proj_geom);

backProj = zeros(vol_geom.GridColCount, vol_geom.GridRowCount, vol_geom.GridSliceCount);

for b = 1:length(proj_geoms)    
    sinogram_id = astra_mex_data3d('create', '-sino', proj_geoms{b}, sinogramData(:,indices(b*2-1):indices(b*2),:));
    astra_mex_data3d('store', recon_id, 0);

    cfg = astra_struct('BP3D_CUDA');
    cfg.ProjectionDataId = sinogram_id;
    cfg.ReconstructionDataId = recon_id;

    alg_id = astra_mex_algorithm('create', cfg);
    astra_mex_algorithm('run', alg_id);

    vol = astra_mex_data3d('get', recon_id);
    backProj = backProj + vol;

    astra_mex_data3d('delete', sinogram_id);
    astra_mex_algorithm('delete', alg_id);
end

astra_mex_data3d('delete', recon_id);

%% Splits up the projection geometry into blocks fitting on GPU
function [proj_geoms, indices] = splitup(proj_geom)

BLOCKSIZE = 200;

% check type of geometry
if strcmp(proj_geom.type(end-2:end), 'vec')
    nbProjections = size(proj_geom.Vectors,1);
else
    nbProjections = length(proj_geom.ProjectionAngles);
end

% compute nb of blocks
blocks = ceil(nbProjections/BLOCKSIZE);
proj_geoms = cell(1,blocks);

indices = zeros(blocks*2);

% create projection geometry per block
for p=1 : blocks
    proj_geoms{p} = proj_geom;
    startline = 1 + (p-1) * BLOCKSIZE;
    endline = min(p*BLOCKSIZE, nbProjections);
    indices( 2*p - 1 : 2*p) = [startline, endline];
    if strcmp(proj_geom.type(end-2:end), 'vec')
        proj_geoms{p}.Vectors = proj_geom.Vectors(startline : endline,:);
    else
        proj_geoms{p}.ProjectionAngles = proj_geom.ProjectionAngles(startline : endline);
    end
end