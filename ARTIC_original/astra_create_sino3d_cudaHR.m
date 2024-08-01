function [sino] = astra_create_sino3d_cudaHR(data, proj_geom, vol_geom, gpu_index)

%--------------------------------------------------------------------------
% [sino] = astra_create_sino_cudaHR(data, proj_geom, vol_geom, gpu_index)
% 
% Create a GPU based forward projection. Works exactly as astra_create_sino_cuda
% but allows for more than 2000 projection angles.
%

% store volume
if (numel(data) > 1)
	if (isa(data,'single'))
		% read-only link
		volume_id = astra_mex_data3d('link','-vol', vol_geom, data, 1);
	else
		volume_id = astra_mex_data3d('create','-vol', vol_geom, data);
	end
else
	volume_id = data;
end

% split up projection geometry in blocks
[proj_geoms, nbProjections, indices] = splitup(proj_geom);

sino = zeros(proj_geom.DetectorColCount, nbProjections, proj_geom.DetectorRowCount);

for b = 1:length(proj_geoms)    
    % store sino
    sino_id = astra_mex_data3d('create','-sino', proj_geoms{b}, 0);

    % create sinogram
    cfg = astra_struct('FP3D_CUDA');
    cfg.ProjectionDataId = sino_id;
    cfg.VolumeDataId = volume_id;
    if nargin > 3
      cfg.option.GPUindex = gpu_index;
    end
    alg_id = astra_mex_algorithm('create', cfg);
    astra_mex_algorithm('iterate', alg_id);

    sino(:,indices(b*2-1):indices(b*2),:) = astra_mex_data3d('get',sino_id);
    astra_mex_data3d('delete',sino_id);
    astra_mex_algorithm('delete', alg_id);
end

if (numel(data) > 1)
    astra_mex_data2d('delete', volume_id);
end

%% Splits up the projection geometry into blocks fitting on GPU
function [proj_geoms, nbProjections, indices] = splitup(proj_geom)

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

indices = zeros(blocks*2,1);

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

