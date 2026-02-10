import React from 'react';

const AnalysisView = ({ imageUrl, isFractured }) => {
    return (
        <div className="relative border-2 border-gray-700 rounded-lg overflow-hidden">
            <h3 className="p-2 bg-gray-800 text-center font-bold">2D X-Ray Analysis</h3>
            
            <img src={imageUrl} alt="X-ray Analysis" className="w-full h-auto" />
            
            {isFractured && (
                <div className="absolute top-0 left-0 w-full h-full pointer-events-none">
                    {/* If using Grad-CAM, the heatmap is already baked into the image. 
                        If using YOLO coordinates, you'd draw a div here. */}
                    <div className="absolute border-4 border-red-500 animate-pulse" 
                         style={{ top: '20%', left: '30%', width: '100px', height: '100px' }}>
                    </div>
                    <span className="absolute top-[15%] left-[30%] bg-red-600 text-white text-xs px-1">
                        FRACTURE DETECTED
                    </span>
                </div>
            )}
        </div>
    );
};

export default AnalysisView;