classdef affine
    properties(Access=private)
        elements;
    end
    properties(SetAccess=private, GetAccess=public)
        nearest = 0;
        bilinear = 1;
        bicubic = 2;
    end
    methods
        function obj = setElements(obj, affine_matrix)
            % The function which sets the elements of an affine matrix.
            % Input
            %   affine_matrix
            %       The matrix which is a 3 by 3 matrix.
            if isequal(size(affine_matrix), [3, 3])
                obj.elements = affine_matrix;
            else
                error('The elements matrix must be 3 X 3!')
            end
        end
        function print_elements(obj)
            % The function which prints the elements of an affine matrix.
            fprintf('The elements of affine matrix is:\n');
            disp(obj.elements);
        end
        function dst = transform(obj, src, interpolation)
            % The function which transforms an image with a affine matrix.
            % Input
            %   src
            %       An input image to be transformed.
            %   interpolation:
            %       Interpolation method which is used in transformation.
            %       Default interpolation is the bilinear.
            % Output
            %   dst
            %       A transformed image.
            if nargin < 3
                interpolation = obj.bilinear;
            end
            switch(interpolation)
                case 0  % Nearest interpolation
                    
                case 1  % Bilinear interpolation
                    
                case 2  % Bicubic interpolation
                    
                otherwise
                    
            end
        end
    end
end