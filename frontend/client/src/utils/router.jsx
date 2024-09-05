import { createBrowserRouter, RouterProvider } from "react-router-dom";
import Home from "../pages/Home";
import Signin from "../pages/Signin/Signin";
import Signup from "../pages/Signup/Signup";
import MainLayout from "../components/Layout/MainLayout";
import ErrorPage404 from "../pages/ErrorPage/Error404";
import VideoUploader from "../pages/Video/VideoUpload";
const router = createBrowserRouter([
    {
        path: '/',
        element: <MainLayout/>,
        // errorElement:  <ErrorPage404/>,
        children: [
            {
                path:'/',
                element:<Home/>
            },
            {
                path: '/signup',
                element: <Signup/>
            },
            {
                path: '/signin',
                element: <Signin/>
            },
            {
                path:'/videos',
                element: <VideoUploader/>
            }
        ]
    }
])

export default function MainRouter(){
    return <RouterProvider router={router}/>
}